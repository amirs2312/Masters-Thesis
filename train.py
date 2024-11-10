import numpy as np
from qiskit import QuantumCircuit
import torch
import matplotlib.pyplot as plt
import pdflatex
import pylatexenc
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# The actual trainnig loop for our Deep Q model

def train_DQN(env, epsilon, epsilon_decay, num_episodes, replay_buffer, batch_size, target_update_freq, epsilon_min, gamma, device, live_model, target_model, criterion, optimizer):


    rewards_list = []
    mean_reward_list = []

    # Enable interactive mode
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.show()

    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0

        # Just in case we reach an optimal policy we want an end criteria for the episode
        for t in range(1, 10000):

            # Epsilon-greedy action selection
            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = live_model(state)
                    action = q_values.max(1)[1].item()
            else:
                action = env.action_space.sample()



            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            total_reward += reward

            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state

            if replay_buffer.size() >= batch_size:

                transitions = replay_buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state).to(device)          
                batch_next_state = torch.cat(batch_next_state).to(device)
                batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float).unsqueeze(1).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float).unsqueeze(1).to(device)


                q_values = live_model(batch_state).gather(1, batch_action)
                with torch.no_grad():
                    max_next_q_values = target_model(batch_next_state).max(1)[0].unsqueeze(1)
                    target_q_values = batch_reward + gamma * max_next_q_values * (1 - batch_done)

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

        # Update target network
        if episode % target_update_freq == 0:
            target_model.load_state_dict(live_model.state_dict())

        # Track rewards
        rewards_list.append(total_reward)
        mean_reward = np.mean(rewards_list[-100:])  # Mean of last 100 episodes
        mean_reward_list.append(mean_reward)

        # Real-time plotting
        ax.clear()
        ax.plot(rewards_list, label='Total Reward per Episode')
        ax.plot(mean_reward_list, label='Mean Reward (Last 100 Episodes)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Classical DQN on CartPole')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Disable interactive mode and show the final plot
    plt.ioff()
    plt.show()





def train_quantum_dqn(env, epsilon, epsilon_decay, num_episodes, replay_buffer, batch_size, 
                        target_update_freq, epsilon_min, gamma, device, live_qmodel, target_qmodel, 
                         criterion, optimizer):
    
    rewards_list = []
    mean_reward_list = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.show()
    
    for episode in range(num_episodes):
        # Reset environment and convert initial state to tensor
        state, info = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0

        if (episode % 10 == 0):
            print("episode %f  Current Mean Reward: %f", episode, mean_reward_list)
        
        # Inner loop for steps within each episode
        for t in range(1, 1000):
            # Epsilon-greedy action selection
            if random.random() > epsilon:
                with torch.no_grad():
                    # Get Q-values from the live quantum model
                    
                    q_values = live_qmodel(state)
                    #print("q_values type:", type(q_values))
                    #p#rint("q_values shape:", q_values.shape)
                    #pint("q_values content:", q_values)
                    action = q_values.max(1)[1].item()
            else:
                action = env.action_space.sample()


            # Step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            total_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state  # Update state for next iteration

            # Training step: sample mini-batch from replay buffer if enough samples are available
            if replay_buffer.size() >= batch_size:
                # Sample a batch and prepare tensors
                transitions = replay_buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                # Convert batches to tensors and ensure compatibility with the device
                batch_state = torch.cat(batch_state).to(device)          
                batch_next_state = torch.cat(batch_next_state).to(device)
                batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float).unsqueeze(1).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float).unsqueeze(1).to(device)

                # Forward pass through the quantum live model for Q-values
                q_values = live_qmodel(batch_state).gather(1, batch_action)

                # Compute target Q-values with the target model
                with torch.no_grad():
                    max_next_q_values = target_qmodel(batch_next_state).max(1)[0].unsqueeze(1)
                    target_q_values = batch_reward + gamma * max_next_q_values * (1 - batch_done)

                # Compute loss and perform backpropagation
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

        # Periodically update the target model
        if episode % target_update_freq == 0:
            target_qmodel.load_state_dict(live_qmodel.state_dict())

        # Track rewards
        rewards_list.append(total_reward)
        mean_reward = np.mean(rewards_list)
        mean_reward_list.append(mean_reward)
    
    # Plotting training progress
        ax.clear()
        ax.plot(rewards_list, label='Total Reward per Episode', color = 'purple')
        ax.plot(mean_reward_list, label='Mean Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Hybrid DQN on CartPole')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Disable interactive mode and show the final plot
    plt.ioff()
    plt.show()