import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter


# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN 모델 정의 (Dueling Network, Noisy Nets 적용)
class Rainbow_DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Rainbow_DQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Value stream (V(s))
        self.value_stream = nn.Linear(128, 1)
        
        # Advantage stream (A(s, a))
        self.advantage_stream = nn.Linear(128, output_dim)
        
        # Noisy layer (Noisy Nets)
        self.noisy_layer = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.feature_layer(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        noisy = self.noisy_layer(x)
        
        # Q(s, a) = V(s) + A(s, a)
        return value + advantage - advantage.mean(dim=1, keepdim=True) + noisy

# 경험 재플레이 버퍼 정의 (우선순위 리플레이)
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, priority=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        probabilities = self._get_probabilities()
        indices = random.choices(range(len(self.buffer)), probabilities, k=batch_size)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        importance_sampling_weights = self._get_importance_sampling_weights(indices)
        
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
            torch.tensor(importance_sampling_weights, dtype=torch.float32).to(device),
            indices
        )
    
    def _get_probabilities(self):
        scaled_priorities = [p ** self.alpha for p in self.priorities]
        total_priority = sum(scaled_priorities)
        return [p / total_priority for p in scaled_priorities]
    
    def _get_importance_sampling_weights(self, indices):
        scaled_priorities = [self.priorities[i] ** self.alpha for i in indices]
        total_priority = sum(scaled_priorities)
        return [(total_priority / len(self.buffer)) / p for p in scaled_priorities]
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Avoid zero priority

    def __len__(self):
        return len(self.buffer)

# DQN 학습 함수
def train_rainbow_dqn(env, episodes=500, model_path="best_model.pth"):
    input_dim = 9
    output_dim = 3
    model = Rainbow_DQN(input_dim, output_dim).to(device)
    target_model = Rainbow_DQN(input_dim, output_dim).to(device)

    # 모델이 저장된 경우, 불러오기
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            target_model.load_state_dict(checkpoint["model_state_dict"])
            epsilon = checkpoint.get("epsilon", 0.105)
            print("Resumed from saved model.")
        except Exception as e:
            print(f"Failed to load model: {e}")

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    buffer = PrioritizedReplayBuffer()
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    update_target_every = 10
    best_score = -float("inf")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state).float().unsqueeze(0).to(device))
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            env.render()

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size)

                q_values = model(states)
                next_q_values = target_model(next_states)

                q_target = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
                q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                loss = (weights * (q_current - q_target.detach()) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update priorities based on errors
                td_errors = (q_current - q_target.detach()).abs().cpu().numpy()
                buffer.update_priorities(indices, td_errors)

            if done:
                print(f"Episode {episode+1}, Score: {env.score}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

                # 최고 점수 저장
                if total_reward > best_score:
                    best_score = total_reward
                    if model_path:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'epsilon': epsilon
                        }, model_path)
                    else:
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "epsilon": epsilon
                        }, "best_play_RainbowDQN.pth")
                    print("New best model saved!")

                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())

