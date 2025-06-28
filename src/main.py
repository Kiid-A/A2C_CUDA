import numpy as np
import time
from algs.traj import Traj
from algs.model import MlpACManual
from algs.a2c import A2C

def simple_game():
    print("Simple game starting...")
    N_parallel = 1
    N_episode = 50
    episode_steps = 20
    obs_dim = 2
    n_actions = 2
    hidden_dim = 128
    ac = MlpACManual(obs_dim, n_actions, hidden_dim)
    traj = Traj(episode_steps)
    trainer = A2C(ac, value_loss_coef=1., actor_loss_coef=1., entropy_coef=0.01, learning_rate=5e-5)
    obs = np.random.randint(0, 10, (N_parallel, obs_dim)).astype(np.float32)
    start_time = time.time()
    for e in range(N_episode):
        for step in range(episode_steps):
            print(f"\rEpisode {e}, Step {step}", end="")
            action, value, actLogProbs = ac.act(obs)
            reward = (action == np.argmax(obs[0])).astype(np.float32) * 2 - 1
            done = np.ones((N_parallel, 1), dtype=np.float32) if e == episode_steps - 1 else np.zeros((N_parallel, 1), dtype=np.float32)
            traj.remember(obs[0], action, reward, actLogProbs, value, done)
            obs = np.random.randint(0, 10, (N_parallel, obs_dim)).astype(np.float32)
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
    simple_game()