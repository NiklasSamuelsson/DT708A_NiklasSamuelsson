from NFQ import NFQ
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    agent = NFQ(
        env=env,
        discount=0.95,
        epsilon=0.1
    )

    agent.train(
        no_episodes=500,
        no_epochs=50
    )

    #agent.play_episodes(
    #    no_episodes=30
    #)
