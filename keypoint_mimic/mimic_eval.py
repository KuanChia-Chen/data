import torch
import argparse
import sys
import pickle
import os

from util.mimic_evaluation_factory import interactive_eval, mimic_camera_eval, receive_internet
from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory, add_env_parser

import threading


if __name__ == "__main__":

    try:
        evaluation_type = sys.argv[1]
        sys.argv.remove(sys.argv[1])
    except:
        raise RuntimeError("Choose evaluation type from ['simple','interactive', or 'no_vis']. Or add a new one.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--slow-factor', default=4, type=int)
    parser.add_argument('--traj-len', default=300, type=int)
    parser.add_argument('--plot-rewards', default=False, action='store_true')
    # Manually handle path argument
    try:
        path_idx = sys.argv.index("--path")
        model_path = sys.argv[path_idx + 1]
        if not isinstance(model_path, str):
            print(f"{__file__}: error: argument --path received non-string input.")
            sys.exit()
    except ValueError:
        print(f"No path input given. Usage is 'python eval.py simple --path /path/to/policy'")

    # model_path = args.path
    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
    critic_checkpoint = torch.load(os.path.join(model_path, 'critic.pt'), map_location='cpu')

    add_env_parser(previous_args_dict['all_args'].env_name, parser, is_eval=True)
    args = parser.parse_args()

    # Overwrite previous env args with current input
    for arg, val in vars(args).items():
        if hasattr(previous_args_dict['env_args'], arg):
            setattr(previous_args_dict['env_args'], arg, val)

    if hasattr(previous_args_dict['env_args'], 'offscreen'):
        previous_args_dict['env_args'].offscreen = True if evaluation_type == 'offscreen' else False
    if hasattr(previous_args_dict['env_args'], 'velocity_noise'):
        delattr(previous_args_dict['env_args'], 'velocity_noise')
    if hasattr(previous_args_dict['env_args'], 'state_est'):
        delattr(previous_args_dict['env_args'], 'state_est')

    # Load environment
    previous_args_dict['env_args'].simulator_type += "_mesh"      # Use mesh model
    #previous_args_dict['env_args'].terrain = 'jay_digit-v3'                    # Change scene here !!!!!!!!!!
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    load_checkpoint(model=critic, model_dict=critic_checkpoint)
    actor.eval()
    actor.training = False

    

    host = '0.0.0.0'  # Listen on all interfaces
    port = 10000
    
    # Start the server thread
    server_thread = threading.Thread(target=receive_internet, args=(host, port))
    server_thread.daemon = True  # Set as a daemon thread so it will close when the main program exits
    server_thread.start()

    if evaluation_type == "mimic":
        mimic_camera_eval(actor=actor, env=env, episode_length_max=args.traj_len)
    elif evaluation_type == 'interactive':
        if not hasattr(env, 'interactive_control'):
            raise RuntimeError("this environment does not support interactive control")
        interactive_eval(actor=actor, env=env, episode_length_max=args.traj_len, critic=critic, plot_rewards=args.plot_rewards)

    server_thread.join()
