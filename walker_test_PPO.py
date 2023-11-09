import gym
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plts
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Humanoid-v2",
                    help='name of Mujoco environement')
args = parser.parse_args()

env = gym.make('Walker2d-v2')
N_A = env.action_space.shape[0]
N_S = env.observation_space.shape[0]

# Actor网络
class Actor(nn.Module):
    def __init__(self, N_S, N_A):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(N_S, 64)
        self.fc2 = nn.Linear(64, 64)
        self.sigma = nn.Linear(64, N_A)
        self.mu = nn.Linear(64, N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # self.set_init([self.fc1,self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal

    # 初始化网络参数
    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))

        mu = self.mu(x)
        log_sigma = self.sigma(x)
        # log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def choose_action(self, s):
        mu, sigma = self.forward(s)
        Pi = self.distribution(mu, sigma)
        return Pi.sample().numpy()


# Critic网洛
class Critic(nn.Module):
    def __init__(self, N_S):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(N_S, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values

# 加载已训练好的模型
actor_path = "C:/Study/Lessons/JQXX/actor_model.path"

actor_net = Actor(N_S, N_A)
actor_net.load_state_dict(torch.load(actor_path))
actor_net.eval()

s = env.reset()
score = 0
while True:
    env.render()
    s = torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0)
    act = actor_net.choose_action(s)[0]
    s_, r, done, info = env.step(act)
    score += r
    s = s_
    if done:
        print('Score:', score)







