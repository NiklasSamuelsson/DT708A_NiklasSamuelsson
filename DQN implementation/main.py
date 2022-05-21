import gym
from DQN import DQN
from ann_designer import ANN

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    in_features = env.reset().shape[0]
    out_features = env.action_space.n

    network = ANN(
        in_features=in_features,
        out_features=out_features,
        lr=0.001
    )

    agent = DQN(
        env=env,
        epsilon=0.1,
        discount=0.99,
        replay_memory_size=10000,
        batch_size=32,
        loss_threshold=0.001,
        reset_target_ANN_updates=1000,
        ANN=network
    )

    agent.train(
        no_episodes=20,
        init_replay_memory=True
    )

    agent.train(
        no_episodes=100,
        init_replay_memory=False
    )