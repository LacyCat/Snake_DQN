import torch
from snake_env import SnakeGameEnv
from train_dqn import DQN

def play_model(model_path="best_model.pth", episodes=5):
    env = SnakeGameEnv(render_mode=True)
    model = DQN(input_dim=6, output_dim=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            env.render()
            with torch.no_grad():
                q_values = model(torch.tensor(state).float().unsqueeze(0))
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                print(f"[Episode {ep+1}] Score: {env.score}, Total Reward: {total_reward:.2f}")
                break