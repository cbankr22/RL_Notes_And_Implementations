'''
Basic DQN outline:

Initialize replay memory D with capacity N
Initialize Q function with random weights
For episode=1, M do:
    Initialize sequence s1 = {x1} and preprocessed sequence p1 = p(s1)
    for t=1, T do
        with probability e select random action a_t
        otherwise select a_t = max_a Q*(p(s_t), a)
        Execute a_t in environment and observe r_t and x_t+1
        Set s_t+1 = s_t, a_t, x_t+1 and preprocess p_t+1 = p(s_t+1)
        Store transition (p_t, a_t, r_t, p_t+1) in D
        Sample minibatch of transitions (p_j, a_j, r_j, p_j+1) from D
        If terminal p_j+1:
            y_j = r_j
        Else:
            y_j = r_j + gamma * max_a' Q*(p_j+1, a')
        Perform a gradient step on (y_j - Q*(p_j, a_j))^2
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym


# Define the Q-Network
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN agent
class DQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_size=5000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = QNet(state_dim, action_dim).float()
        self.target_net = QNet(state_dim, action_dim).float()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return torch.argmax(self.q_net(torch.FloatTensor(state))).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.q_net(states).gather(1, actions)
        print(actions)
        print(self.q_net(states))
        print(current_q)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# Training
env = gym.make('CartPole-v1')
agent = DQN(env.observation_space.shape[0], env.action_space.n)

episodes = 300
target_update = 10
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    while True:
        if type(state) == tuple:
            state = state[0]
        action = agent.select_action(state)
        next_state, reward, done, _,  _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward
        state = next_state
        if done:
            break

    if episode % target_update == 0:
        agent.update_target_network()
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()