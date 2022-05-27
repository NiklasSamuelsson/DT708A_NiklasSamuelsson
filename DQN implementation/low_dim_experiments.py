import gym
from DQN import DQN
from ann_designer import SimpleMLP
import pandas as pd
import numpy as np
import time

# Prepare environment
env = gym.make("CartPole-v1")

# Define non-varying settings
in_features = env.reset().shape[0]
out_features = env.action_space.n
high_dim_input = False
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.9999
discount = 0.99
replay_memory_size = 100000
no_init_eps = 500
max_no_eps = 2000

# Define experiment parameters
experiment_name = "experiment 2"
no_trials = 10
learing_rates = [0.0005, 0.001, 0.002]
synch_invervals = [100, 200, 400]
batch_sizes = [16, 32, 64]

# Define logging
path = "C:/Users/niksa/Projects/DT708A Reinforcement learning part 2/DT708A_NiklasSamuelsson/DQN implementation/Experiment results/"
summary_path = path + "/" + experiment_name + "/" + experiment_name + "_low dim_summary.csv"
details_path = path + "/" + experiment_name + "/" + experiment_name + "_low dim_details_"  # More to be added later
summary = pd.DataFrame(
    columns=[
        "high_dim_input",
        "epsilon_max",
        "epsilon_min",
        "epsilon_decay",
        "discount",
        "replay_memory_size",
        "no_init_episodes",
        "max_no_episodes",
        "no_trials",
        "learning_rate",
        "synch_interval",
        "batch_size",
        "average_no_ep_solved",
        "variance_no_ep_solved",
        "min_no_ep_solved",
        "max_no_ep_solved"
    ]
)
non_varying = [
    high_dim_input,
    epsilon,
    epsilon_min,
    epsilon_decay,
    discount,
    replay_memory_size,
    no_init_eps,
    max_no_eps,
    no_trials
]


# Run experiment
tot_iterations = len(learing_rates) * len(synch_invervals) * len(batch_sizes) * no_trials
curr_iteration = 1
for lr in learing_rates:
    for si in synch_invervals:
        for bs in batch_sizes:
            print("STARTING NEW CONFIG")
            print("Learning rate:", lr)
            print("Synch interval:", si)
            print("Batch size:", bs)
            train_hist = []
            test_hist = []
            for t in range(no_trials):
                print("Starting iteration", curr_iteration, "out of", tot_iterations)
                start = time.time()

                ANN = SimpleMLP(
                    in_features=in_features,
                    out_features=out_features,
                    lr=lr
                )

                agent = DQN(
                    env=env,
                    high_dim_input=high_dim_input,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    discount=discount,
                    replay_memory_size=replay_memory_size,
                    batch_size=bs,
                    reset_target_ANN_updates=si,
                    ANN=ANN
                )

                train, test, eval = agent.train(
                    no_episodes=no_init_eps,
                    init_replay_memory=True
                )

                train, test, eval = agent.train(
                    no_episodes=max_no_eps,
                    init_replay_memory=False
                )

                train_hist.append(train)
                test_hist.append(test)

                # Calculate eval statistics per trial
                eval_results = pd.DataFrame(
                    columns=[
                        "episode",
                        "eval_avg",
                        "eval_var",
                        "eval_min",
                        "eval_max"
                    ]
                )
                
                for eval_episode, eval_trial in enumerate(eval):
                    eval_avg = np.mean(eval_trial)
                    eval_var = np.var(eval_trial)
                    eval_min = np.min(eval_trial)
                    eval_max = np.max(eval_trial)

                    eval_episode *= 50

                    res = [
                        eval_episode,
                        eval_avg,
                        eval_var,
                        eval_min,
                        eval_max
                    ]
                    eval_results.loc[len(eval_results)] = res
                eval_path = details_path + "eval_" + str(lr) + "_" + str(si) + "_" + str(bs) + "_" + str(t) + ".csv"
                eval_results.to_csv(eval_path, index=False, sep=";")

                end = time.time()
                print("Elapsed time trial", str(t+1) + ":", end-start)
                curr_iteration += 1
            
            # Calculate aggregated statistics
            train_stats = []
            for i in range(len(train_hist)):
                train_stats.append(len(train_hist[i]))

            # Statistics on no episodes until solved or reached max no episodes
            solved_avg = np.mean(train_stats)
            solved_var = np.var(train_stats)
            solved_min = np.min(train_stats)
            solved_max = np.max(train_stats)

            varying = [
                lr,
                si,
                bs,
                solved_avg,
                solved_var,
                solved_min,
                solved_max
            ]

            res = non_varying + varying
            summary.loc[len(summary)] = res
            summary.to_csv(summary_path, index=False, sep=";")

            # Calculate statistics for each episode
            stats_per_ep = pd.DataFrame(
                columns=[
                    "episode",
                    "train_avg",
                    "train_var",
                    "train_min",
                    "train_max",
                    "test_avg",
                    "test_var",
                    "test_min",
                    "test_max"
                ]
            )

            # Loop over the shortest list (first trial to solve it) to be able to save data in a good way
            for i in range(min(len(x) for x in train_hist)):
                train_temp = []
                test_temp = []
                for j in range(len(train_hist)):
                    train_temp.append(train_hist[j][i])
                    test_temp.append(test_hist[j][i])

                train_avg = np.mean(train_temp)
                train_var = np.var(train_temp)
                train_min = np.min(train_temp)
                train_max = np.max(train_temp)

                test_avg = np.mean(test_temp)
                test_var = np.var(test_temp)
                test_min = np.min(test_temp)
                test_max = np.max(test_temp)

                res = [
                    i,
                    train_avg,
                    train_var,
                    train_min,
                    train_max,
                    test_avg,
                    test_var,
                    test_min,
                    test_max
                ]

                stats_per_ep.loc[len(stats_per_ep)] = res

            details_path_temp = details_path + "train test_" + str(lr) + "_" + str(si) + "_" + str(bs) + ".csv"
            stats_per_ep.to_csv(details_path_temp, index=False, sep=";")

            
                

