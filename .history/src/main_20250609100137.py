import torch
import time
from traj import Traj
from model import MlpAC
from a2c import A2C

def simple_game():
    print("Simple game starting...")
    N_parallel = 1
    N_episode = 5000
    episode_steps = 20
    obs_dim = 2
    n_actions = 2
    hidden_dim = 128
    ac = MlpAC(obs_dim, n_actions, hidden_dim)
    traj = Traj(episode_steps)
    trainer = A2C(ac, 1., 1., 0.1, 1e-3)
    obs = torch.randint(0, 10, (N_parallel, obs_dim)).float()
    start_time = time.time()
    for e in range(N_episode):
        for step in range(episode_steps):
            print(f"\rEpisode {e}, Step {step}", end="")
            with torch.no_grad():
                action, value, actLogProbs = ac.act(obs)
            reward = (action == obs.argmax(1).unsqueeze(1)).float() * 2 - 1
            done = torch.ones((N_parallel, 1)) if e == episode_steps - 1 else torch.zeros((N_parallel, 1))
            traj.remember(obs[0], action[0], reward[0], actLogProbs[0], value[0], done[0])
            obs = torch.randint(0, 10, (N_parallel, obs_dim)).float()
        print("\r\n")
        print(f"Episode {e} begins to update:")
        train_info = trainer.update(traj)
        print(f"Episode {e} finished, Train info:")
        for key, value in train_info.items():
            print(f"{key}: {value}")
        traj.clear()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration} seconds.")

if __name__ == "__main__":
    torch.set_num_threads(12)
    simple_game()