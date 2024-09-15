import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from tqdm import tqdm


env = gym.make("CartPole-v1")
env.metadata["render_fps"] = 1000000
BATCHSIZE = 100
GAMMA = 0.99
EPISODES = 1000000

state, _ = env.reset()


class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stack = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.stack(x)


class Network:
    def __init__(self) -> None:
        self.model = DQN()
        self.targetModel = DQN()
        self.targetModel.load_state_dict(self.model.state_dict())
        self.steps = 0
        self.memory = deque(maxlen=100_000)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.ai = 0
        self.rand = 0
        self.rewardPerGame = []
        self.rewardPerGame.append(0)

    def train(self, state, newState, action, reward, done):
        self.memory.append((state, newState, action, reward, done))
        self.steps += 1

        if len(self.memory) < BATCHSIZE:
            return

        batch = random.sample(self.memory, BATCHSIZE)
        states, new_states, actions, rewards, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        new_states = torch.tensor(new_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.targetModel(new_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (~dones)

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 100 == 0:
            self.targetModel.load_state_dict(self.model.state_dict())

    def getAction(self, state):
        self.epsilon = 0.0001 + 0.9 * np.exp(1e-4 * -self.steps)
        if np.random.random() < self.epsilon:
            self.rand += 1
            return env.action_space.sample()
        else:
            self.ai += 1
            with torch.no_grad():
                return (
                    self.model(torch.tensor(state, dtype=torch.float32)).argmax().item()
                )


ngames = 0
network = Network()
for i in tqdm(range(EPISODES)):
    while True:
        action = network.getAction(state)
        nextState, reward, terminated, truncated, info = env.step(action)
        network.train(state, nextState, action, 0 if terminated else reward, terminated)
        network.rewardPerGame[ngames] += reward
        state = nextState

        if truncated or terminated:
            # print(
            #     f"Games: {ngames}, AI: {network.ai}, Random: {network.rand}, Percentage: {round(network.ai / (network.ai + network.rand) * 100.0)}, Reward: {int(network.rewardPerGame[ngames])}, Steps: {network.steps}, Epsilon: {round(network.epsilon, 3)}"
            # )
            ngames += 1
            network.ai = 0
            network.rand = 0
            network.rewardPerGame.append(0)
            state, info = env.reset()
            break
    if network.epsilon <= 0.01:
        break

env.close()
env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
while True:
    action = network.getAction(state)
    nextState, reward, terminated, truncated, info = env.step(action)
    network.train(state, nextState, action, 0 if terminated else reward, terminated)
    network.rewardPerGame[ngames] += reward
    state = nextState
    if truncated or terminated:
        print(
            f"Games: {ngames}, AI: {network.ai}, Random: {network.rand}, Percentage: {round(network.ai / (network.ai + network.rand) * 100.0)}, Reward: {int(network.rewardPerGame[ngames])}, Steps: {network.steps}, Epsilon: {round(network.epsilon, 3)}"
        )
        ngames += 1
        network.ai = 0
        network.rand = 0
        network.rewardPerGame.append(0)
        state, info = env.reset()
