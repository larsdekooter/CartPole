import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import sys


env = gym.make("CartPole-v1", render_mode="human")
BATCHSIZE = 10

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
        self.random = 0
        self.ai = 0

    def train(self, state, newState, action, reward, done):
        # Add experience to memory
        self.memory.append((state, newState, action, reward, done))

        # Only train if we have enough samples
        if len(self.memory) < BATCHSIZE:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, BATCHSIZE)

        # Unpack the batch
        states, newStates, actions, rewards, dones = map(np.array, zip(*batch))

        # Convert to tensors
        states = torch.FloatTensor(states)
        newStates = torch.FloatTensor(newStates)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute Q values
        currentQValues = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        nextQValues = self.targetModel(newStates).max(1)[0].detach()

        # Compute target Q values
        targetQValues = rewards + (1 - dones) * 0.99 * nextQValues

        # Compute loss
        loss = self.criterion(currentQValues, targetQValues)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.steps % 100 == 0:
            self.targetModel.load_state_dict(self.model.state_dict())

        self.steps += 1

    def getAction(self, state):
        e = 0.0001 + 0.9 * np.exp(1e-6 * self.steps)
        self.steps += 1
        if np.random.random() < e:
            self.random += 1
            return env.action_space.sample()
        self.ai += 1
        with torch.no_grad():
            return self.model(torch.tensor(state, dtype=torch.float32)).argmax().item()


ngames = 0
network = Network()
while True:
    action = network.getAction(state)
    nextState, reward, terminated, truncated, info = env.step(action)
    network.train(state, nextState, action, 0 if terminated else reward, terminated)

    if truncated or terminated:
        ai = network.ai
        rand = network.random
        network.ai = 0
        network.random = 0

        ngames += 1

        state, info = env.reset()
        print(f"{round(ai / (ai + rand) * 100.0)}")
