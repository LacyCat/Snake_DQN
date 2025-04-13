import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dueling DQN 모델 정의
class Hybrid_DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Hybrid_DQN, self).__init__()
        # Feature extraction layers
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Scalar output representing value
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Output dimensions for actions
        )

    def forward(self, x):
        # Extract features
        feature = self.feature(x)
        
        # Value and advantage
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)
        
        # Q-value = Value + Advantage
        return value + (advantage - advantage.mean())

# 경험 재플레이 버퍼 정의
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# DQN 학습 함수
def train_hybrid_dqn(env, episodes=500, model_path="best_model.pth"):
    input_dim = 9
    output_dim = 3
    model = Hybrid_DQN(input_dim, output_dim).to(device)
    target_model = Hybrid_DQN(input_dim, output_dim).to(device)

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

    buffer = ReplayBuffer()
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
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                q_values = model(states)
                next_q_values = target_model(next_states)

                q_target = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
                q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                loss = criterion(q_current, q_target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                        }, "best_play_Hybrid_DQN.pth")
                    print("New best model saved!")

                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
