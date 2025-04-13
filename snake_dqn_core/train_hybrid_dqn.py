import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
import numpy as np

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dueling DQN 모델 정의
class Hybrid_DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Hybrid_DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# Prioritized Experience Replay 버퍼 정의
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, epsilon=1e-5):
        self.buffer = deque(maxlen=capacity)
        self.alpha = alpha  # 우선순위 강도
        self.beta = beta    # 중요도 샘플링 비율
        self.epsilon = epsilon  # 작은 우선순위 값 처리
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize for stability
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
            torch.tensor(weights, dtype=torch.float32).to(device),
            indices
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)

# DQN 학습 함수
def train_hybrid_dqn(env, episodes=500, model_path=None, debug=False):
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

    buffer = PrioritizedReplayBuffer()
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    update_target_every = 10
    best_score = -float("inf")

    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

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

                loss = (weights * criterion(q_current, q_target.detach())).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 우선순위 업데이트
                td_errors = (q_current - q_target.detach()).abs().cpu().numpy()
                buffer.update_priorities(indices, td_errors)

            if done:
                if debug:
                    loss_val = loss.item() if 'loss' in locals() else 0.0
                    print(f"Episode {episode+1}, Score: {env.score}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, "
                        f"Gamma: {gamma}, Batch Size: {batch_size}, Eps Min: {epsilon_min}, Eps Decay: {epsilon_decay:.4f}, "
                        f"Loss: {loss_val:.4f}")
                else:
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
        scheduler.step()

        if (episode + 1) % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
