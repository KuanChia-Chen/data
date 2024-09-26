import numpy as np
import sys
import termios
import time
import torch
import cv2
#from util.xbox import XboxController
from util.keyboard import Keyboard
from util.colors import OKGREEN, FAIL, WARNING, ENDC
from util.reward_plotter import Plotter
import socket
import json

received_data = []

def receive_internet(host, port):
    global received_data
    
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the host and port
    s.bind((host, port))
    
    # Listen for incoming connections
    s.listen(1)
    
    print(f"Server listening on {host}:{port}")


    while True:

        # Accept a connection
        conn, addr = s.accept()
        #print(f"Connected by {addr}")
        
        # Receive data from the client
        data = conn.recv(1024)
        
        if not data:
            break
        
        received_data.append(json.loads(data))
        
        conn.close()


def mimic_control(action, env):

    global received_data
    
    movement = []

    while received_data:
        movement = received_data.pop(0)

    env.step(action)

def mimic_camera_eval(actor, env, episode_length_max=300, critic=None, plot_rewards=False):
    """Simply evaluating policy in visualization window with user input, interactive_xbox_eval

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print(f"Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds.{ENDC}")
    keyboard = Keyboard()

    global received_data
    

    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        env.sim.viewer_init(fps = env.default_policy_rate)
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()

        # initial state [r_z y x, l_z y x, h]
        movement = [0.15, 0.2, 0.3, 0.15, 0.2, 0.3, 0.8]

        while render_state:

            start_time = time.time()

            # receive movement point
            while received_data:
                movement = received_data.pop(0)
                #print(movement)
                
            cmd = None
            if keyboard.data():
                cmd = keyboard.get_input()
            if not env.sim.viewer_paused():
                
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                state, reward, done, infos = env.step(action,np.array(movement))
                episode_length += 1
                episode_reward.append(reward)
                
                if critic is not None:
                    if hasattr(env, 'get_privilege_state'):
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                    # print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                env.viewer_update_cop_marker()
            if cmd is not None:
                env.interactive_control(cmd)
            if cmd == "r":
                done = True
            env.display_controls_menu()
            env.display_control_commands()
            render_state = env.sim.viewer_render()
            delaytime = max(0, 1/env.default_policy_rate - (time.time() - start_time))
            time.sleep(delaytime)
            
            if done:
                state = env.reset(interactive_evaluation=True)
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                env.display_control_commands()
                episode_length = 0
                episode_reward = []
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                done = False
        keyboard.restore()
        # clear terminal on ctrl+q
        print(f"\033[{len(env.control_commands_dict) + 3 - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)



def interactive_eval(actor, env, episode_length_max=300, critic=None, plot_rewards=False):
    """Simply evaluating policy in visualization window with user input, interactive_xbox_eval

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print(f"Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds.{ENDC}")
    keyboard = Keyboard()

    if plot_rewards:
        plotter = Plotter()

    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()
        env.sim.viewer_init(fps = env.default_policy_rate)
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()
        while render_state:
            start_time = time.time()
            cmd = None
            if keyboard.data():
                cmd = keyboard.get_input()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                state, reward, done, infos = env.step(action)
                episode_length += 1
                episode_reward.append(reward)
                if plot_rewards:
                    plotter.add_data(infos, done or cmd == "quit")
                if critic is not None:
                    if hasattr(env, 'get_privilege_state'):
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                    # print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                env.viewer_update_cop_marker()
            if cmd is not None:
                env.interactive_control(cmd)
            if cmd == "r":
                done = True
                env.display_controls_menu()
                env.display_control_commands()
            render_state = env.sim.viewer_render()
            delaytime = max(0, 1/env.default_policy_rate - (time.time() - start_time))
            time.sleep(delaytime)
            
            if done:
                state = env.reset(interactive_evaluation=True)
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                env.display_control_commands()
                episode_length = 0
                episode_reward = []
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                done = False
        keyboard.restore()
        # clear terminal on ctrl+q
        print(f"\033[{len(env.control_commands_dict) + 3 - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)