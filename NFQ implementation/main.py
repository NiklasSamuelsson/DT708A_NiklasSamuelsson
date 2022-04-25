from NFQ import NFQ
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    agent = NFQ(
        env=env,
        discount=0.95,
        epsilon=0.1
    )

    # Generate randomly controlled experience
    agent.train(
        no_episodes=1000,
        no_epochs=20
    )

    agent.refit_Q(no_epochs=10, batch_size=100)

    agent.play_episodes(no_episodes=20)

    #agent.play_episodes(
    #    no_episodes=30
    #)
