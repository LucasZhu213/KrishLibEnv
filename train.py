from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from collections import deque
import numpy as np
import random
from game import Game
from PIL import Image

import multiprocessing
from env import MotionProfilePacman # Or whatever your class is

def test_spawn():
    try:
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=lambda: print("Success!"))
        p.start()
        p.join()
        print("Subprocess created successfully.")
    except Exception as e:
        print(f"Subprocess failed: {e}")

class Worker():
    def __init__(self, worker_id, update_queue, transition_queue, model_state):
        self.worker_id = worker_id
        self.update_queue = update_queue
        self.transition_queue = transition_queue
        self.init_state = model_state

    def work(self):
        self.env = make_vec_env(
            "MotionProfilePacman-v1",
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        self.DQN = DQN(
            "MultiInputPolicy",
            self.env,
            exploration_fraction=0.5,
            exploration_final_eps=0.05,
            verbose=1,
            learning_rate=5e-5,
            buffer_size=1,
            tensorboard_log="tensorboard",
            gamma=0.9
        )
        self.DQN.policy.load_state_dict(self.init_state)
        obs = self.env.reset()
        while True:
            while not self.update_queue.empty():
                model_state = self.update_queue.get()
                self.update_dqn(model_state)
            action, _ = self.DQN.predict(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.transition_queue.put((
                obs,
                next_obs,
                action,
                reward,
                done,
                info
            ))
            obs = next_obs
            if done[0]:
                obs = self.env.reset()

    def update_dqn(self, model_state):
        self.DQN.policy.load_state_dict(model_state)


register(
    id="MotionProfilePacman-v1",
    entry_point="env:MotionProfilePacman",
    max_episode_steps=300,
)

if __name__ == "__main__":
    # # env = gym.make('MotionProfilePacman-v1',render_mode="human")
    # #
    # # obs = env.reset()
    # #
    # # for i in range(1000):
    # #     action = random.choice(range(5))
    # #     obs, reward, terminated, something, info = env.step(action)
    # #     if i == 500:
    # #         img = Image.fromarray(obs)
    # #         img.save("last_frame.png")
    # #     if terminated:
    # #         print(terminated)
    # #         obs = env.reset()

    # env = make_vec_env(
    #     "MotionProfilePacman-v1",
    #     n_envs=8,
    #     # env_kwargs={"render_mode": "human"},
    #     vec_env_cls=DummyVecEnv,
    # )

    # model = PPO(
    #     "MultiInputPolicy", env, device="cpu", verbose=1, tensorboard_log="tensorboard"
    # )  # default policy is "MlpPolicy"
    # model.learn(total_timesteps=int(1e4), log_interval=4)
    # model.save("ppo_pacbot")

    # del model  # remove to demonstrate saving and loading

    # model = PPO.load("ppo_pacbot")

    # env = make_vec_env(
    #     "MotionProfilePacman-v1",
    #     n_envs=1,
    #     # env_kwargs={"render_mode": "human"},
    #     vec_env_cls=DummyVecEnv,
    # )

    env = make_vec_env("MotionProfilePacman-v1", n_envs=1, vec_env_cls=DummyVecEnv)
    # policy_kwargs = dict(net_arch=[256, 256])
    model = DQN(
        "MultiInputPolicy", 
        env, 
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=1, 
        learning_rate=5e-5,
        buffer_size=50000, 
        # exploration_fraction=0.1,
        tensorboard_log="tensorboard",
        gamma=0.9
    )
    model.set_logger(
        configure(
            "tensorboard",
            ["stdout", "tensorboard"],
        )
    )
    #model.learn(total_timesteps=int(1e6))

    env = make_vec_env(
        "MotionProfilePacman-v1",
        n_envs=1,
        env_kwargs={"render_mode": "human"},
        vec_env_cls=DummyVecEnv,
    )
    
    #obs = env.reset()
    # action = np.array([0])
    # obs, reward, terminated, truncated, info = env.step(action)

    transition_queue = multiprocessing.Queue(
        maxsize=10000
    )
    update_queues = []
    workers = []
    model_state = {
        k: v.detach().cpu().clone()
        for k, v in model.policy.state_dict().items()
    }
    for i in range(8):
        upd_queue = multiprocessing.Queue(maxsize=10000)
        tmp_worker = Worker(i, update_queue=upd_queue, transition_queue=transition_queue, model_state=model_state)
        p = multiprocessing.Process(target=tmp_worker.work)
        update_queues.append(upd_queue)
        p.start()
        workers.append(p)
    batch_size = 64
    max_transitions = 100000
    count = 0
    while count <= max_transitions:
        transition = transition_queue.get()
        model.replay_buffer.add(*transition); count += 1
        if model.replay_buffer.size() >= batch_size:
            model.train(
                gradient_steps=1,
                batch_size=batch_size,
            )
            model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.policy.state_dict().items()
            }
            #model.logger.dump(step=count)
        for uq in update_queues:
            uq.put(model_state)
    for worker in workers:
        worker.terminate()
    for worker in workers:
        worker.join()
    model.save("dqn_pacbot")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done[0]:
            print("Episode finished!")
            obs = env.reset()
        # if terminated or truncated:
        #     print(terminated)
        #     obs = env.reset()
    