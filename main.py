import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from utils.Gradanalyzer import GradientAnalyzer
from utils.ReDo import GradientReDo,ReDo
from SAC.configs import parse_args




def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Creates a function that initializes a Gymnasium environment with specific parameters.
    
    Args:
        env_id (str): Identifier for the Gymnasium environment
        seed (int): Random seed for reproducibility
        idx (int): Index for the environment instance
        capture_video (bool): Whether to record videos of the environment
        run_name (str): Name of the current run for video saving
        
    Returns:
        callable: A function that creates and returns the configured environment
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """
    Implements the Q-network for Soft Actor-Critic algorithm with configurable architecture.
    
    Features:
    - Configurable network width and depth through multipliers
    - Multiple activation function options
    - Handles continuous action spaces
    
    Args:
        env (gym.Env): Gymnasium environment instance
        activation (str): Activation function type ('relu', 'leaky_relu', 'tanh', etc.)
        width_multiplier (float): Multiplier for layer widths (default: 1)
        depth_multiplier (float): Multiplier for network depth (default: 1)
    """
    def __init__(self, env, activation, width_multiplier=1, depth_multiplier=1):
        super().__init__()
        self.base_width = 256
        self.base_depth = 2
        hidden_width = int(self.base_width * width_multiplier)
        hidden_depth = int(self.base_depth * depth_multiplier)
        
        # Input layer: Combines state and action
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            hidden_width,
        )
        
        # Hidden layers: Dynamic depth based on multiplier
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_depth - 1): 
            self.hidden_layers.append(nn.Linear(hidden_width, hidden_width))
        
        # Output layer: Single Q-value
        self.fc_out = nn.Linear(hidden_width, 1)

        
        # Activation function configuration
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=args.negative_slope)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "silu":
            self.activation = nn.SiLU() 

            

    def forward(self, x, a):
        """
        Forward pass of the Q-network.
        
        Args:
            x (torch.Tensor): State input
            a (torch.Tensor): Action input
            
        Returns:
            torch.Tensor: Q-value for the state-action pair
        """
        x = torch.cat([x, a], dim=1)
        x = self.activation(self.fc1(x))
        
        for layer in self.hidden_layers:
            x_before = x
            x_after = layer(x_before)
            x = self.activation(x_after)  
        
        x = self.fc_out(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """
    Implements the policy network for Soft Actor-Critic algorithm.
    
    Features:
    - Outputs both mean and log standard deviation for action distribution
    - Uses reparameterization trick for backpropagation
    - Handles continuous action spaces with proper scaling
    
    Args:
        env (gym.Env): Gymnasium environment instance
    """
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        
        # Action scaling for proper bounds
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        """
        Forward pass to compute action distribution parameters.
        
        Args:
            x (torch.Tensor): State input
            
        Returns:
            tuple: (mean, log_std) for the action distribution
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        """
        Samples actions using the reparameterization trick.
        
        Args:
            x (torch.Tensor): State input
            
        Returns:
            tuple: (action, log_prob, mean) where:
                - action: Sampled action from the policy
                - log_prob: Log probability of the action
                - mean: Mean action (without sampling)
        """
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with action bounds correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parse_args()

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=args.filtered_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs,args.activation,args.width_multiplier,args.depth_multiplier).to(device)
    qf2 = SoftQNetwork(envs,args.activation,args.width_multiplier,args.depth_multiplier).to(device)
    qf1_target = SoftQNetwork(envs,args.activation,args.width_multiplier,args.depth_multiplier).to(device)
    qf2_target = SoftQNetwork(envs,args.activation,args.width_multiplier,args.depth_multiplier).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)


    q_analyzer = GradientAnalyzer(qf1)


    if args.exp_name == "redo":
        q1_redo = ReDo(qf1,tau=args.redo_tau,frequency=args.redo_frequency,use_lecun_init=args.redo_use_lecun_init)
        q2_redo = ReDo(qf2,tau=args.redo_tau,frequency=args.redo_frequency,use_lecun_init=args.redo_use_lecun_init)

    if args.exp_name == "grad_redo":
        q1_grad_redo = GradientReDo(qf1,tau=args.grad_redo_tau,frequency=args.grad_redo_frequency,use_lecun_init=args.grad_use_lecun_init)
        q2_grad_redo = GradientReDo(qf2,tau=args.grad_redo_tau,frequency=args.grad_redo_frequency,use_lecun_init=args.grad_use_lecun_init)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=[args.seed + i for i in range(envs.num_envs)])
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step=global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step=global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()


            if global_step % args.grad_analyzer_frequency == 0:
                ratio,_,zero_grad_ratio = q_analyzer.analyze_gradients((data.observations,data.actions))
                writer.add_scalar("charts/ratio", ratio, global_step=global_step)
                writer.add_scalar("charts/zero_grad_ratio", zero_grad_ratio, global_step=global_step)

            if args.exp_name == "redo":
                info = q1_redo.step((data.observations,data.actions))
                q2_redo.step((data.observations,data.actions))
                if "dormant_fraction" in info:
                    writer.add_scalar("charts/reset_fraction", info["dormant_fraction"], global_step=global_step)

            if args.exp_name == "grad_redo":
                info = q1_grad_redo.step()
                q2_grad_redo.step()
                if "dormant_fraction" in info:
                    writer.add_scalar("charts/reset_fraction", info["dormant_fraction"], global_step=global_step)

            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step=global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step=global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step=global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step=global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step=global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step=global_step)
                writer.add_scalar("losses/alpha", alpha, global_step=global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step=global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step=global_step)

    envs.close()
    writer.close()