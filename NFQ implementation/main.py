from NFQ import NFQ
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    agent = NFQ(
        env=env,
        discount=0.90,
        epsilon=0.1)

    agent.train(
        no_episodes=100,
        no_epochs=20
    )

    agent.play_episodes(
        no_episodes=30
    )
