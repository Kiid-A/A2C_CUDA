import numpy as np
import time
from matplotlib import pyplot as plt
from algs.traj import Traj
from algs.model import MlpACManual, MlpACTorch
from algs.a2c_torch import A2CTorch
from algs.a2c import A2C

def simple_game():
    history = {}
    history['rewards'] = []


    print("Simple game starting...")
    N_parallel = 1
    N_episode = 10000
    episode_steps = 20
    obs_dim = 2
    n_actions = 2
    hidden_dim = 128

    # ac = MlpACManual(obs_dim, n_actions, hidden_dim)
    # trainer = A2C(ac, value_loss_coef=0.5, actor_loss_coef=1., entropy_coef=0.0, learning_rate=2e-5)
    ac = MlpACTorch(obs_dim, n_actions, hidden_dim)
    trainer = A2CTorch(ac, value_loss_coef=0.5, actor_loss_coef=1., entropy_coef=1e-3, learning_rate=2e-5)

    traj = Traj(episode_steps)
    obs = np.random.randint(0, 10, (N_parallel, obs_dim)).astype(np.float32)
    start_time = time.time()
    for e in range(N_episode):
        for step in range(episode_steps):
            print(f"\rEpisode {e}, Step {step}", end="")
            action, value, actLogProbs = ac.act(obs)
            reward = (action == np.argmax(obs[0])).astype(np.float32)
            done = np.ones((N_parallel, 1), dtype=np.float32) if e == episode_steps - 1 else np.zeros((N_parallel, 1), dtype=np.float32)
            traj.remember(obs[0], action, reward, actLogProbs, value, done)
            obs = np.random.randint(0, 10, (N_parallel, obs_dim)).astype(np.float32)
        print("\r\n")
        print(f"Episode {e} begins to update:")
        train_info = trainer.update(traj)
        print(f"Episode {e} finished, Train info:")
        for key, value in train_info.items():
            # print(f"{key}: {value}")
            if not (key in history):
                history[key] = []
            history[key].append(value)
        episode_rewards = traj.get_rewards()
        assert len(episode_rewards) > 0
        history["rewards"].append(np.sum(episode_rewards))
        traj.clear()
        if (e % 2000 == 0) or (e == N_episode - 1):
            # plot to ./key_name.png, y=value, x=episode
            for key in history.keys():
                plt.plot(history[key])
                plt.title(key)
                plt.savefig(f"./{key}.png")
                plt.clf()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration} seconds.")

if __name__ == "__main__":
    simple_game()