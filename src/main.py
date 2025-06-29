import numpy as np
import time
from matplotlib import pyplot as plt
from algs.traj import Traj
from algs.model import MlpACManual, MlpACTorch
from algs.a2c_torch import A2CTorch
from algs.a2c import A2C
trainer_args_th = dict(value_loss_coef=1, actor_loss_coef=1, entropy_coef=1e-4, learning_rate=6e-5)

def simple_game():
    history = {}
    history['rewards'] = []
    # 添加计时相关变量
    total_rollout_time = 0
    total_update_time = 0
    total_time = 0

    print("Simple game starting...")
    N_parallel = 1
    N_episode = 10
    episode_steps = 20
    obs_dim = 2
    n_actions = 2
    hidden_dim = 1280
    trainer_args = dict(value_loss_coef=1, actor_loss_coef=1, entropy_coef=1e-4, learning_rate=8e-4)
    ac = MlpACManual(obs_dim, n_actions, hidden_dim, cpu=False); trainer = A2C(ac, **trainer_args)
    # ac = MlpACTorch(obs_dim, n_actions, hidden_dim); trainer = A2CTorch(ac, **trainer_args_th)

    traj = Traj(episode_steps)
    obs = np.random.randint(0, 2, (N_parallel, obs_dim)).astype(np.float32)
    start_time = time.time()
    for e in range(N_episode):
        if e == 1: start_time = time.time()
        # Rollout阶段计时
        rollout_start = time.time()
        for step in range(episode_steps):
            print(f"\rEpisode {e}, Step {step}", end="")
            action, value, actLogProbs = ac.act(obs)
            reward = (action == np.argmax(obs[0])).astype(np.float32)
            done = np.ones((N_parallel, 1), dtype=np.float32) if e == episode_steps - 1 else np.zeros((N_parallel, 1), dtype=np.float32)
            traj.remember(obs[0], action, reward, actLogProbs, value, done)
            obs = np.random.randint(0, 2, (N_parallel, obs_dim)).astype(np.float32)
        rollout_time = time.time() - rollout_start
        if e != 0: total_rollout_time += rollout_time
        
        print("\r\n")
        print(f"Episode {e} begins to update:")
        # Update阶段计时
        update_start = time.time()
        train_info = trainer.update(traj)
        update_time = time.time() - update_start
        if e != 0: total_update_time += update_time
        
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
        if (e % 500 == 0) or (e == N_episode - 1):
            # plot to ./key_name.png, y=value, x=episode
            for key in history.keys():
                plt.plot(history[key])
                plt.title(key)
                try:
                    plt.savefig(f"./src/result/{key}.png")
                except Exception:
                    plt.savefig(f"./result/{key}.png")
                plt.clf()
        
        total_time = time.time() - start_time
        # 计算并打印三个指标
        print(f"SPE = {total_time/(e + 1)}")  # 总体速度
        print(f"MRT = {total_rollout_time/(e + 1)}")  # Rollout阶段平均时间
        print(f"MUT = {total_update_time/(e + 1)}")  # Update阶段平均时间

    # ac.free_intermediate()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration} seconds.")
    # 打印最终统计结果
    print(f"Final SPE: {duration/N_episode}")
    print(f"Final MRT: {total_rollout_time/N_episode}")
    print(f"Final MUT: {total_update_time/N_episode}")

if __name__ == "__main__":
    simple_game()