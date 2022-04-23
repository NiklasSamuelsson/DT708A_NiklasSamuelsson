import gym
import random

env = gym.make("CartPole-v1")

state = env.reset()

# Access the image
state = env.render(mode="rgb_array")
print(state.shape)
print(state[200][300])

no_episodes = 50
for e in range(no_episodes):
    reward = 0
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = random.choice([0,1])
        state, d_reward, done, _ = env.step(action)
        reward += d_reward
