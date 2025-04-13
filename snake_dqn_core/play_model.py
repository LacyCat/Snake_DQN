import torch
from snake_dqn_core.snake_env import SnakeGameEnv
from snake_dqn_core.train_dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def play_model(model_path="best_model.pth", episodes=5):
    env = SnakeGameEnv(render_mode=True)
    model = DQN(input_dim=9, output_dim=3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            env.render()
            with torch.no_grad():
                q_values = model(torch.tensor(state).float().unsqueeze(0).to(device))
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                print(f"[Episode {ep+1}] Score: {env.score}, Total Reward: {total_reward:.2f}")
                break