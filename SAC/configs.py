import argparse
from distutils.util import strtobool

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, choices=["baseline", "redo", "grad_redo"],
                         default="gradual_unfreeze",
                         help="the name of this experiment, choose from: baseline, redo, gradient_redo")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="number of environments to run in parallel")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Humanoid",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Humanoid-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000,
        help="total timesteps of the experiments (will be multiplied by num_envs internally)")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size (should account for num_envs)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning (will be multiplied by num_envs internally)")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    

    # ReDo specific arguments
    parser.add_argument("--redo-frequency", type=int, default=1000,
        help="the frequency of checking and resetting dormant neurons")
    parser.add_argument("--redo-tau", type=float, default=0,
        help="the threshold for resetting dormant neurons")
    parser.add_argument("--redo-use-lecun-init", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use LeCun initialization for resetting neurons")

    
    # Grad_redo specific arguments
    parser.add_argument("--grad-redo-frequency", type=int, default=1000,
        help="the frequency of checking and resetting dormant neurons")
    parser.add_argument("--grad-use-lecun-init", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use LeCun initialization for resetting neurons")
    parser.add_argument("--grad-redo-tau", type=float, default=0,
        help="the threshold for resetting dormant neurons")

        

    # Activation function specific arguments
    parser.add_argument("--activation", type=str, default="relu",
        help="the activation function to use, choose from: leaky_relu, relu, tanh, gelu")
    parser.add_argument("--negative-slope", type=float, default=0.01,
        help="negative slope for leaky_relu activation function")

    # Grad_analyzer specific arguments
    parser.add_argument("--grad-analyzer-frequency", type=int, default=1000,
        help="the frequency of checking and resetting dormant neurons")
    


    parser.add_argument("--depth-multiplier", type=int, default=1,
        help="the depth for the hidden layers")
    parser.add_argument("--width-multiplier", type=int, default=1,
        help="the width for the hidden layers")
    



        

    args = parser.parse_args()
    
    # Create a dictionary to store the actually used parameters
    config_dict = vars(args).copy()
    
    # Remove irrelevant parameters based on experiment type
    if args.exp_name != "redo":
        # Remove ReDo specific parameters
        config_dict.pop("redo_frequency", None)
        config_dict.pop("redo_tau", None)
        config_dict.pop("redo_use_lecun_init", None)
        config_dict.pop("redo_reset_steps", None)

    if args.exp_name != "grad_redo":
        # Remove Grad_ReDo specific parameters
        config_dict.pop("grad_redo_frequency", None)
        config_dict.pop("grad_use_lecun_init", None)
        config_dict.pop("grad_redo_tau", None)
        config_dict.pop("grad_redo_reset_steps", None)

    
    # Save the filtered configuration dictionary back to args
    args.filtered_config = config_dict
    
    # fmt: on
    return args


