import gym
from DQN import DQN
from ann_designer import ANN

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    in_features = env.reset().shape[0]
    out_features = env.action_space.n

    # TODO: try clipping loss function
    network = ANN(
        in_features=in_features,
        out_features=out_features,
        lr=0.001
    )

    # TODO: try CER (https://arxiv.org/pdf/1712.01275.pdf)
    # TODO: try double Q-learning
    agent = DQN(
        env=env,
        epsilon=1,
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        discount=0.99,
        replay_memory_size=100000,
        batch_size=64,
        loss_threshold=0.1,
        reset_target_ANN_updates=1000,
        ANN=network
    )

    agent.train(
        no_episodes=500,
        init_replay_memory=True
    )

    agent.train(
        no_episodes=800,
        init_replay_memory=False
    )

    for i in range(20):
        agent.play_one_episode(
            render=True,
            verbose=True
        )