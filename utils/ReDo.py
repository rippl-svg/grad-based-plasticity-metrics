from functools import partial
import math
from typing import Dict, List, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class BaseReDo:
    """
    Base class for ReDo (Reset of Dormant units) implementations.
    
    ReDo is a technique to maintain neural network plasticity by identifying and
    reinitializing dormant neurons during training. This base class provides common
    functionality for different ReDo variants.
    
    Features:
    - Configurable neuron dormancy detection
    - Support for both Kaiming and LeCun initialization
    - Optimizer state management for smooth training
    - Flexible reset scheduling
    
    Args:
        model (nn.Module): The neural network model to apply ReDo to
        tau (float): Threshold for determining dormant neurons (default: 0)
        use_lecun_init (bool): Whether to use LeCun initialization instead of Kaiming (default: False)
        frequency (int): How often to check and reset neurons (default: 1000)
        optimizer (optim.Adam, optional): Optimizer for resetting moments
        reset_steps (int, optional): Maximum steps to perform resets. None means reset indefinitely
    """
    def __init__(self, model: nn.Module, tau: float = 0, use_lecun_init: bool = False, 
                 frequency: int = 1000, optimizer: optim.Adam = None, reset_steps: int = None):
        assert tau >= 0, "tau must be non-negative"
        self.model = model
        self.tau = tau
        self.use_lecun_init = use_lecun_init
        self.current_step = 0
        self.frequency = frequency
        self.optimizer = optimizer
        self.reset_steps = reset_steps

    @staticmethod
    def _kaiming_uniform_reinit(layer: Union[nn.Linear, nn.Conv2d], mask: torch.Tensor) -> None:
        """
        Reinitializes selected neurons using Kaiming uniform initialization.
        
        Args:
            layer (Union[nn.Linear, nn.Conv2d]): Layer containing neurons to reinitialize
            mask (torch.Tensor): Boolean mask indicating which neurons to reinitialize
        """
        fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
        gain = nn.init.calculate_gain(nonlinearity="relu", param=math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        
        # Reset weights for masked neurons
        with torch.no_grad():
            layer.weight.data[mask, ...] = torch.empty_like(
                layer.weight.data[mask, ...]
            ).uniform_(-bound, bound)

            # Reset bias if present
            if layer.bias is not None:
                if isinstance(layer, nn.Conv2d):
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        layer.bias.data[mask, ...] = torch.empty_like(
                            layer.bias.data[mask, ...]
                        ).uniform_(-bound, bound)
                else:
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    layer.bias.data[mask, ...] = torch.empty_like(
                        layer.bias.data[mask, ...]
                    ).uniform_(-bound, bound)

    @staticmethod
    def _lecun_normal_reinit(layer: Union[nn.Linear, nn.Conv2d, nn.LayerNorm], mask: torch.Tensor) -> None:
        """
        Reinitializes selected neurons using LeCun normal initialization.
        
        Args:
            layer (Union[nn.Linear, nn.Conv2d]): Layer containing neurons to reinitialize
            mask (torch.Tensor): Boolean mask indicating which neurons to reinitialize
        """
        if isinstance(layer, nn.LayerNorm):
            layer.weight.data[mask] = torch.ones_like(layer.weight.data[mask])
            layer.bias.data[mask] = torch.zeros_like(layer.bias.data[mask])
            return

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        variance = 1.0 / fan_in
        stddev = math.sqrt(variance) / 0.87962566103423978
        
        # Reset weights
        with torch.no_grad():
            layer.weight[mask] = nn.init._no_grad_trunc_normal_(
                layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0
            )
            layer.weight[mask] *= stddev
            
            # Reset bias if it exists
            if layer.bias is not None:
                layer.bias.data[mask] = 0.0

    def _reset_adam_moments(self, reset_masks) -> None:
        """
        Resets the Adam optimizer's moment estimates for dormant neurons.
        
        This ensures smooth continuation of training after neuron reinitialization
        by preventing momentum from immediately pulling weights back to their
        previous values.
        
        Args:
            reset_masks (List[torch.Tensor]): List of boolean masks for each layer
        """
        assert isinstance(self.optimizer, optim.Adam), "Moment resetting currently only supported for Adam optimizer"
        # Gets a list of all parameters (assuming all parameters are in the first parameter group)
        params = self.optimizer.param_groups[0]['params']
        
        for layer_idx, mask in enumerate(reset_masks):
            try:
                # Get parameter indices for current layer
                weight_idx = 2 * layer_idx
                bias_idx = 2 * layer_idx + 1
                next_weight_idx = 2 * (layer_idx + 1)

                # Reset moments for weights
                weight_param = params[weight_idx]
                weight_state = self.optimizer.state[weight_param]
                weight_state["exp_avg"][mask, ...] = 0.0
                weight_state["exp_avg_sq"][mask, ...] = 0.0
                weight_state["step"] = 0

                # Reset moments for bias if present
                if bias_idx < len(params):
                    bias_param = params[bias_idx]
                    if bias_param in self.optimizer.state:
                        bias_state = self.optimizer.state[bias_param]
                        bias_state["exp_avg"][mask] = 0.0
                        bias_state["exp_avg_sq"][mask] = 0.0
                        bias_state["step"] = 0

                # Reset moments for outgoing connections
                if next_weight_idx < len(params):
                    next_weight_param = params[next_weight_idx]
                    next_weight_state = self.optimizer.state[next_weight_param]
                    
                    # Handle conv to linear layer transition
                    if len(weight_state["exp_avg"].shape) == 4 and len(next_weight_state["exp_avg"].shape) == 2:
                        num_repetition = next_weight_state["exp_avg"].shape[1] // mask.shape[0]
                        linear_mask = torch.repeat_interleave(mask, num_repetition)
                        next_weight_state["exp_avg"][:, linear_mask] = 0.0
                        next_weight_state["exp_avg_sq"][:, linear_mask] = 0.0
                    else:
                        # Standard case (same layer types)
                        next_weight_state["exp_avg"][:, mask, ...] = 0.0
                        next_weight_state["exp_avg_sq"][:, mask, ...] = 0.0
                    next_weight_state["step"] = 0

            except (IndexError, KeyError) as e:
                print(f"Warning: Layer {layer_idx} parameter not found in optimizer state")
                continue

    def _reset_dormant_neurons(self, model: nn.Module, redo_masks: List[torch.Tensor]) -> nn.Module:
        """Re-initializes the weights of dormant neurons."""
        # Only get Conv2d and Linear layers
        layers = [(name, layer) for name, layer in model.named_modules() 
                 if isinstance(layer, (nn.Conv2d, nn.Linear))]
        
        
        assert len(redo_masks) == len(layers) - 1, (
            f"Number of masks ({len(redo_masks)}) must match number of layers-1 ({len(layers)-1})"
        )

        # Reset ingoing weights
        with torch.no_grad():
            for i in range(len(layers)-1):
                mask = redo_masks[i]
                layer = layers[i][1]
                next_layer = layers[i + 1][1]

                # Skip if no dead neurons
                if torch.all(~mask):
                    continue

                # Reset weights using specified initialization
                if self.use_lecun_init:
                    self._lecun_normal_reinit(layer, mask)
                else:
                    self._kaiming_uniform_reinit(layer, mask)

        return model

    def _get_redo_masks(self, *args, **kwargs) -> List[torch.Tensor]:
        """Abstract method to compute masks for dormant neurons."""
        raise NotImplementedError("Subclasses must implement _get_redo_masks")

    def step(self, *args, **kwargs) -> Dict[str, any]:
        """Abstract method to perform ReDo step."""
        raise NotImplementedError("Subclasses must implement step")


class ReDo(BaseReDo):
    """Activation-based ReDo implementation (baseline method)."""
    
    def _get_activation_and_reset(self, layer_name: str, layer: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> Callable:
        """Get activation values and immediately reset the current layer's hook function"""
        def hook(layer: Union[nn.Linear, nn.Conv2d, nn.LayerNorm], 
                input: Tuple[torch.Tensor], 
                output: torch.Tensor) -> None:
            # Get activation values
            activation = F.relu(output)
            
            # Compute score for current layer
            if isinstance(layer, nn.Conv2d):  # Conv layer
                score = activation.abs().mean(dim=(0, 2, 3))
            elif isinstance(layer, nn.Linear):  # Linear layer
                score = activation.abs().mean(dim=0)
            elif isinstance(layer, nn.LayerNorm):
                score = activation.abs().mean(dim=0)
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
            
            # Compute mask according to different modes
            if self.mode == 'threshold':
                # Normalize score
                normalized_score = score / (score.mean() + 1e-9)
                # Create mask (True for dormant neurons)
                layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
                if self.tau > 0.0:
                    layer_mask[normalized_score <= self.tau] = 1
                else:
                    layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
            
            elif self.mode == 'percentage':
                # Select least active neurons by percentage
                k = max(1, int(len(score) * self.percentage))
                threshold = torch.kthvalue(score, k).values if k < len(score) else torch.min(score)
                layer_mask = score <= threshold
            
            elif self.mode == 'hybrid':
                # First filter by threshold, then limit max reset ratio
                normalized_score = score / (score.mean() + 1e-9)
                threshold_mask = normalized_score <= self.tau
                
                k_max = max(1, int(len(score) * self.max_percentage))
                if threshold_mask.sum() > k_max:
                    combined_score = score.clone()
                    combined_score[~threshold_mask] = float('inf')
                    _, indices = torch.topk(combined_score, k_max, largest=False)
                    layer_mask = torch.zeros_like(score, dtype=torch.bool)
                    layer_mask[indices] = True
                else:
                    layer_mask = threshold_mask
            
            # Statistics
            self.layer_neurons += layer_mask.numel()
            self.layer_dormant += layer_mask.sum().item()
            
            # Immediately reset current layer
            if torch.any(layer_mask) and self.should_reset:
                # Reset weights using specified initialization
                if self.use_lecun_init:
                    self._lecun_normal_reinit(layer, layer_mask)
                else:
                    self._kaiming_uniform_reinit(layer, layer_mask)
                
                # Reset optimizer state (if any)
                if self.optimizer is not None:
                    # Find this layer's param in optimizer
                    for param_group in self.optimizer.param_groups:
                        for param in param_group['params']:
                            if param is layer.weight:
                                state = self.optimizer.state[param]
                                if 'exp_avg' in state:
                                    state['exp_avg'][layer_mask] = 0.0
                                    state['exp_avg_sq'][layer_mask] = 0.0
                            elif param is layer.bias and layer.bias is not None:
                                state = self.optimizer.state[param]
                                if 'exp_avg' in state:
                                    state['exp_avg'][layer_mask] = 0.0
                                    state['exp_avg_sq'][layer_mask] = 0.0
                
        return hook

    @torch.inference_mode()
    def step(self, obs: torch.Tensor) -> Dict[str, any]:
        """Perform layerwise ReDo step"""
        self.current_step += 1
        
        # Check if in reset range
        in_reset_range = True
        if self.reset_steps is not None:
            in_reset_range = self.current_step >= self.reset_steps[0] and self.current_step <= self.reset_steps[1]
            
        self.should_reset = self.current_step % self.frequency == 0 and in_reset_range
        
        if self.should_reset:
            # Initialize statistics
            self.layer_neurons = 0
            self.layer_dormant = 0
            
            # Register hooks
            handles = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                    handles.append(
                        module.register_forward_hook(
                            self._get_activation_and_reset(name, module)
                        )
                    )

            # Get activations and reset (hook function will handle)
            if isinstance(obs, tuple):
                _ = self.model(*obs)
            else:
                _ = self.model(obs)

            # Compute statistics
            dormant_fraction = (self.layer_dormant / max(1, self.layer_neurons)) * 100
            
            print(f"Re-initializing dormant neurons")
            print(f"Total neurons: {self.layer_neurons} | "
                  f"Dormant neurons: {self.layer_dormant} | "
                  f"Dormant fraction: {dormant_fraction:.2f}%")

            # Remove hooks
            for handle in handles:
                handle.remove()

            return {
                "dormant_fraction": dormant_fraction,
                "dormant_count": self.layer_dormant,
                "total_neurons": self.layer_neurons
            }
        else:
            return {}


class GradientReDo(BaseReDo):
    """Gradient-based ReDo implementation with cosine annealing (our method)."""
    def __init__(self, model: nn.Module, mode: str = 'threshold', tau: float = 0, 
                     percentage: float = 1, max_percentage: float = 1,
                     use_lecun_init: bool = False, frequency: int = 1000, 
                     optimizer: optim.Adam = None, reset_steps: List[int] = None):
        super().__init__(model, tau, use_lecun_init, frequency, optimizer, reset_steps)
        self.mode = mode
        self.percentage = percentage / 100 if 0 <= percentage <= 100 else 0.01
        self.max_percentage = max_percentage / 100 if 0 <= max_percentage <= 100 else 0.01
        
    def _get_layer_mask(self, layer: nn.Module, tau: float) -> torch.Tensor:
        """Compute mask for dormant neurons in a single layer"""
        if layer.weight.grad is None:
            # Handle no gradient case
            if isinstance(layer, nn.Conv2d):
                return torch.zeros(layer.out_channels, dtype=torch.bool, device=layer.weight.device)
            elif isinstance(layer, nn.Linear):
                return torch.zeros(layer.out_features, dtype=torch.bool, device=layer.weight.device)
                
        # Compute mean absolute value of gradients
        if isinstance(layer, nn.Conv2d):
            grad_magnitude = layer.weight.grad.abs().mean(dim=(1, 2, 3))
        elif isinstance(layer, nn.Linear):
            grad_magnitude = layer.weight.grad.abs().mean(dim=1)
        elif isinstance(layer, nn.LayerNorm):
            grad_magnitude = layer.weight.grad.abs()
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # Handle logic for different modes
        if self.mode == 'threshold':
            normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
            mask = normalized_grad <= tau
        elif self.mode == 'percentage':
            k = max(1, int(len(grad_magnitude) * self.percentage))
            threshold = torch.kthvalue(grad_magnitude, k).values if k < len(grad_magnitude) else torch.min(grad_magnitude)
            mask = grad_magnitude <= threshold
        elif self.mode == 'hybrid':
            normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
            threshold_mask = normalized_grad <= tau
            
            k_max = max(1, int(len(grad_magnitude) * self.max_percentage))
            if threshold_mask.sum() > k_max:
                combined_grad = grad_magnitude.clone()
                combined_grad[~threshold_mask] = float('inf')
                _, indices = torch.topk(combined_grad, k_max, largest=False)
                mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                mask[indices] = True
            else:
                mask = threshold_mask
                
        return mask
    
    def _reset_single_layer(self, layer: nn.Module, mask: torch.Tensor) -> None:
        """Reset dormant neurons in a single layer"""
        if torch.all(~mask):
            return
            
        with torch.no_grad():
            # Reset weights using specified method
            if self.use_lecun_init:
                self._lecun_normal_reinit(layer, mask)
            else:
                self._kaiming_uniform_reinit(layer, mask)
                
            # Reset gradients
            if layer.weight.grad is not None:
                layer.weight.grad[mask] = 0.0
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad[mask] = 0.0
                
        # If optimizer exists, reset corresponding optimizer state for params
        if self.optimizer is not None:
            self._reset_layer_optimizer_state(layer, mask)
    
    def _reset_layer_optimizer_state(self, layer: nn.Module, mask: torch.Tensor) -> None:
        """Reset optimizer state for a single layer"""
        if not isinstance(self.optimizer, optim.Adam):
            return
            
        # Find corresponding param in optimizer state
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param is layer.weight:
                    state = self.optimizer.state[param]
                    if 'exp_avg' in state:
                        state['exp_avg'][mask] = 0.0
                        state['exp_avg_sq'][mask] = 0.0
                elif param is layer.bias and layer.bias is not None:
                    state = self.optimizer.state[param]
                    if 'exp_avg' in state:
                        state['exp_avg'][mask] = 0.0
                        state['exp_avg_sq'][mask] = 0.0
    
    @torch.no_grad()
    def step(self) -> Dict[str, any]:
        """Perform layerwise gradient ReDo step"""
        self.current_step += 1
        
        # Check if in reset range
        in_reset_range = True
        if self.reset_steps is not None:
            in_reset_range = self.current_step >= self.reset_steps[0] and self.current_step <= self.reset_steps[1]
            
        if self.current_step % self.frequency == 0 and in_reset_range:
            # Initialize statistics
            total_neurons = 0
            dormant_count = 0
            
            # Get all resettable layers
            layers = [(name, layer) for name, layer in self.model.named_modules() 
                     if isinstance(layer, (nn.Linear, nn.Conv2d, nn.LayerNorm))]
            
            # Process layer by layer
            for name, layer in layers:
                    
                # Compute mask for current layer
                mask = self._get_layer_mask(layer, self.tau)
                
                # Statistics
                layer_neurons = mask.numel()
                layer_dormant = mask.sum().item()
                total_neurons += layer_neurons
                dormant_count += layer_dormant
                
                # Reset current layer
                if layer_dormant > 0:
                    self._reset_single_layer(layer, mask)
            
            # Compute overall statistics
            dormant_fraction = (dormant_count / max(1, total_neurons)) * 100
            
            print(f"Gradient reset complete")
            print(f"Total neurons: {total_neurons} | "
                  f"Dormant neurons: {dormant_count} | "
                  f"Dormant fraction: {dormant_fraction:.2f}%")
            
            return {
                "dormant_fraction": dormant_fraction,
                "dormant_count": dormant_count,
                "total_neurons": total_neurons,
            }
        else:
            return {}
