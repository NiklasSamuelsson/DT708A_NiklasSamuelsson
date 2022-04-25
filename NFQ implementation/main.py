from NFQ import NFQ
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    agent = NFQ(
        env=env,
        discount=0.95,
        epsilon=0.1
    )

    # Unfortunately, the agent is not learning anything
    agent.train(
        no_episodes=100,
        no_epochs=15,
        batch_size=1000,
        no_steps=1000,
        random_action=False
    )

    #agent.refit_Q(no_epochs=10, batch_size=100)

    #agent.play_episodes(no_episodes=20)

