import gym
from DQN import DQN
from ann_designer import SimpleMLP, CNN, ImageMLP

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    _ = env.reset()
    in_channels = env.render(mode="rgb_array").shape[2]
    in_channels = 4
    out_features = env.action_space.n

    # TODO: automate testing

    network = CNN(
        in_channels=in_channels,
        out_features=out_features,
        lr=0.001
    )

    # TODO: try CER (https://arxiv.org/pdf/1712.01275.pdf)
    # TODO: try double Q-learning
    agent = DQN(
        env=env,
        high_dim_input=True,
        epsilon=1,
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        discount=0.99,
        replay_memory_size=35000,
        batch_size=32,
        reset_target_ANN_updates=1000,
        ANN=network
    )

    agent.train(
        no_episodes=50,
        init_replay_memory=True
    )

    agent.train(
        no_episodes=5000,
        init_replay_memory=False
    )

    for i in range(10):
        agent.play_one_episode(
            render=True,
            verbose=True
        )