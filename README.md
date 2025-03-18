# Neural Network Plasticity Metrics with SAC

This repository implements various neural network plasticity metrics in the context of Soft Actor-Critic (SAC) reinforcement learning algorithm. It features novel approaches to maintain network plasticity during training, including ReDo (Reset of Dormant units) and Gradient-based ReDo implementations.
## Project Structure

```
plasticity_metrics/
├── main.py              # Main training script
├── sac.py              # SAC algorithm implementation
├── utils/
│   ├── ReDo.py         # ReDo/our ReDo implementations
│   ├── Gradanalyzer.py # Gradient analysis tools for our metrics
│   └── L2RegularizationWithNullspace.py
├── SAC/
│   └── configs.py      # Configuration parameters
├── run_experiments.sh  # Batch experiment script
└── requirements.txt    # Dependencies
```
## Features

- **Soft Actor-Critic Implementation**
  - Configurable network architectures (width and depth)
  - Multiple activation function support
  - Automatic entropy tuning
  - Continuous action space support

- **Plasticity Metrics**
  - ReDo: Activation-based neuron reset
  - Gradient ReDo: Gradient-based neuron reset
  - Adaptive L2 regularization with nullspace analysis
  - Comprehensive gradient analysis tools

- **Experiment Management**
  - Weights & Biases integration
  - TensorBoard support
  - Configurable hyperparameters
  - Multi-seed experiment support
## Code Logic for Key Components for ReDo-style reset & plasticity metrics (Introduce of the most important coding part, @johan, maybe a simplify way is just transfer the related code module to your DQN/DT project)
### 
### To integrate all the Rest schedule functionality into other projects (e.g., DQN/DT), focus on these core components from `ReDo.py`:

#### 1. Base Reset Functionality (`BaseReDo` class)
- `_reset_dormant_neurons()`: Core reset logic for reinitializing weights
- `_kaiming_uniform_reinit()` and `_lecun_normal_reinit()`: Weight initialization methods
- `_reset_adam_moments()`: Optional optimizer state management

#### 2. **Baseline** Activation-based Rest (`ReDo` class) 
Key methods to adapt:
- `_get_activation()`: Hook for capturing layer activations
- `_get_redo_masks()`: Logic for identifying dormant neurons
- `step()`: Main control flow for reset timing

#### 3. **Our** Gradient-based Rest (`GradientReDo` class)
Key methods to adapt:
- `_get_redo_masks()`: Gradient-based dormancy detection
- `step()`: Simplified reset control flow

#### Integration Steps:
1. Copy relevant initialization methods from `BaseReDo`
2. Choose either activation or gradient-based approach
3. Adapt mask generation logic to your network architecture
4. Integrate reset calls into your training loop
5. Optional: combine optimizer state management with ReDo (This technique is used to mitigate plasticity loss isolately, which would interfere with real performance comparison of redo and gradient-based redo)

The core reset functionality is model-agnostic and can be adapted to any neural network architecture with minimal changes.

### Understanding the implementation of our grad-based plasticity metrics (in code it is called nullspace ratio) with `Gradanalyzer.py`:

The `GradientAnalyzer` class provides tools to measure neural network plasticity through gradient analysis. Key components:

1. **Gradient Collection** (Implemented in `_register_grad_hooks()` and `_build_params_index()`)
   - Hooks are registered to collect gradients from all model parameters during initialization
   - Uses a pre-allocated buffer (`grad_buffer`) sized to total parameter count for efficient storage
   - Maintains parameter metadata index for mapping gradients back to layers
   - Automatically handles both single-tensor and multi-tensor parameters
   
2. **(Our plasticity metrics) Nullspace Ratio Calculation**
   - Located in `analyze_gradients()` method in Gradanalyzer.py (*noticed that for AC architechure we only measure the critic*)
   - Execution pipeline:
     1. Sample parameters hierarchically across layers (sample scale can be changed, in SAC we set to 1000 to make pipline fast)
     2. Compute per-sample gradients using `_compute_batch_gradients()`:
        - Clear gradient buffer for each sample
        - Do backward pass with gradient_outputs mask
        - Collect gradients into grad_buffer
     3. Normalize gradient vectors by L2 norm
     4. Perform SVD decomposition on gradient matrix:
        - U: Left singular vectors (batch_size × batch_size)
        - S: Singular values in descending order 
        - Vh: Right singular vectors transposed
     5. Calculate nullspace ratio:
        - Find singular values below threshold (tol * largest singular value, tol is also can be changed empirically)
        - Ratio = number of small values in S metrix / batch size
     6. Also computes zero vector ratio as supplementary metric
   - Returns nullspace ratio (0-1) indicating plasticity loss
      - Higher ratio = More redundant/ineffective gradient directions
      - Lower ratio = Better gradient diversity and learning capacity
   
3. **Layer Contribution Analysis (not used, TO de done in the future maybe)**
   - Maps gradient importance back to model layers
   - Provides insights into which layers contribute most to plasticity loss

This analysis helps identify when networks need intervention via ReDo resets.

## Usage

### Basic Training

Run a single experiment with default settings:
```bash
python main.py --env-id HalfCheetah-v4 --exp-name baseline
```

### Experiment Types (we suggest to use the run_experiments.sh)

1. **Baseline SAC**:
```bash
python main.py --env-id HalfCheetah-v4 --exp-name baseline --activation relu
```

2. **ReDo Implementation**:
```bash
python main.py --env-id HalfCheetah-v4 --exp-name redo --activation relu --redo-tau 0.1
```

3. **Gradient ReDo**:
```bash
python main.py --env-id HalfCheetah-v4 --exp-name grad_redo --activation relu --grad-redo-tau 0.1
```

### Batch Experiments

Run multiple experiments with different configurations:
```bash
bash run_experiments.sh
```

This script will run experiments with:
- Multiple seeds
- Different activation functions
- Various network architectures
- All plasticity methods

### Configuration Options

Key parameters that can be modified:

- Network Architecture: (for scaling test)
  - `--width-multiplier`: Adjust network width (default: 1)
  - `--depth-multiplier`: Adjust network depth (default: 1)

- Training Parameters:
  - `--learning-starts`: Steps before starting training (default: 100)
  - `--batch-size`: Batch size for training (default: 256)
  - `--gamma`: Discount factor (default: 0.99)

- Plasticity Parameters:
  - `--redo-tau`: Threshold for ReDo (default: 0)
  - `--grad-redo-tau`: Threshold for Gradient ReDo (default: 0)
  - `--redo-frequency`: Reset frequency (default: 1000)

- Logging:
  - `--track`: Enable W&B logging
  - `--wandb-project-name`: W&B project name
  - `--capture-video`: Record environment videos

## Monitoring and Visualization

### Weights & Biases
Enable W&B logging with the `--track` flag. Monitored metrics include:
- Episode returns and lengths
- Q-values and losses
- Plasticity metrics (dormant neuron ratio, gradient based ratio, zero_grad ratio)
- Network gradients statistics  

### TensorBoard
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir runs
```

## Main Training Loop Logic

The main training loop in `main.py` demonstrates how the plasticity metrics and reset mechanisms are integrated into the SAC training process. Here's a detailed breakdown:

### Initialization Phase
```python
# Initialize gradient analyzer for Q-function
q_analyzer = GradientAnalyzer(qf1)

# Initialize ReDo or Gradient ReDo based on experiment type
if args.exp_name == "redo":
    q1_redo = ReDo(qf1, tau=args.redo_tau, frequency=args.redo_frequency, use_lecun_init=args.redo_use_lecun_init)
    q2_redo = ReDo(qf2, tau=args.redo_tau, frequency=args.redo_frequency, use_lecun_init=args.redo_use_lecun_init)

if args.exp_name == "grad_redo":
    q1_grad_redo = GradientReDo(qf1, tau=args.grad_redo_tau, frequency=args.grad_redo_frequency, use_lecun_init=args.grad_use_lecun_init)
    q2_grad_redo = GradientReDo(qf2, tau=args.grad_redo_tau, frequency=args.grad_redo_frequency, use_lecun_init=args.grad_use_lecun_init)
```

### Training Loop Integration

1. **Gradient Analysis (Every N Steps)**
   ```python
   if global_step % args.grad_analyze_freq == 0:
       # Analyze gradients for plasticity metrics
       nullspace_ratio, zero_grad_ratio = q_analyzer.analyze_gradients()
       # Log metrics to W&B
       if args.track:
           wandb.log({"nullspace_ratio": nullspace_ratio, "zero_grad_ratio": zero_grad_ratio})
   ```

2. **ReDo Reset (Based on Frequency)**
   ```python
   if args.exp_name == "redo":
       # Activation-based reset
       q1_redo.step()
       q2_redo.step()
   elif args.exp_name == "grad_redo":
       # Gradient-based reset
       q1_grad_redo.step()
       q2_grad_redo.step()
   ```

3. **Training Step Integration**
   ```python
   # Regular SAC training step
   qf1_loss, qf2_loss = train_critic(...)
   
   # After critic update, check for plasticity loss
   if args.exp_name == "grad_redo":
       # Additional gradient analysis after training
       nullspace_ratio, zero_grad_ratio = q_analyzer.analyze_gradients()
   ```

### Key Points:

1. **GradientAnalyzer**:
   - Initialized once at the start
   - Called periodically to monitor network plasticity
   - Provides nullspace ratio and zero gradient ratio metrics

2. **ReDo**:
   - Called based on frequency parameter
   - Resets neurons based on activation patterns
   - Applied to both Q-functions independently

3. **GradientReDo**:
   - Similar frequency-based calling pattern
   - Uses gradient information for reset decisions
   - Also applied to both Q-functions

4. **Integration Timing**:
   - Analysis happens every `grad_analyze_freq` steps
   - Resets occur based on `redo_frequency` or `grad_redo_frequency`
   - Metrics are logged to W&B when tracking is enabled

This integration ensures continuous monitoring of network plasticity while maintaining the core SAC training process.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 