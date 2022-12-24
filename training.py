import os
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses

from helpers import fixed, two_mini_random, update_target
from agent import Coop_MADDPG


def run(
    env,
    continuous,
    total_trials=1,
    total_episodes=100,
    disc_actions_num=2,
    seed=1453,
    buffer_capacity=50000,
    batch_size=64,
    num_agents=4,
    learn_step=25,
    std_dev=0.3,
    epsilon=0.2,
    actor_lr=0.002,
    critic_lr=0.003,
    clip=1,
    gamma=0.99,
    tau=0.005,
    adam_eps=1e-07,
    amsgrad=False,
    theta=0.15,
    gamma_func=fixed,
    tau_func=fixed,
    critic_lr_func=fixed,
    actor_lr_func=fixed,
    std_dev_func=fixed,
    epsilon_func=fixed,
    reward_mod=False,
    start_steps=0,
    loss_func=losses.MeanAbsoluteError(),
    mean_number=20,
    solved=999,
    render=False,
    weights_directory="Weights/",
    plots_directory="Graphs/",
    output=True,
    total_time=True,
    use_gpu=True,
    return_data=False,
):

    start_time = time.time()

    # _ = env.seed(seed)
    rng = np.random.default_rng(seed)

    num_states = env.observation_space[0].shape[0]
    if continuous:
        num_actions = env.action_space.shape[0]
    else:
        num_actions = 1

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Normalize action space according to https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    # env.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype='float32')

    ep_reward_list = []
    avg_reward_list = []
    true_reward_list = []
    true_avg_reward_list = []

    for trial in range(total_trials):
        step = 0

        # Add sublists for each trial
        avg_reward_list.append([])
        ep_reward_list.append([])
        true_reward_list.append([])
        true_avg_reward_list.append([])

        agent = Coop_MADDPG(
            num_states,
            num_actions,
            num_agents,
            continuous,
            buffer_capacity,
            batch_size,
            std_dev,
            epsilon,
            actor_lr,
            critic_lr,
            gamma,
            tau,
            clip,
            adam_eps,
            amsgrad,
            theta,
            disc_actions_num,
            loss_func,
        )

        for ep in range(total_episodes):
            before = time.time()

            agent.gamma = gamma_func(agent.gamma, ep)
            agent.tau = tau_func(agent.tau, ep)
            agent.critic_lr = critic_lr_func(agent.critic_lr, ep)
            agent.actor_lr = actor_lr_func(agent.actor_lr, ep)
            agent.std_dev = std_dev_func(agent.std_dev, ep)
            agent.epsilon = epsilon_func(agent.epsilon, ep)

            prev_state = env.reset()
            episodic_reward = np.zeros(num_agents)
            true_reward = np.zeros(num_agents)

            while True:
                if render:
                    env.render()

                tf_prev_state = tf.convert_to_tensor(prev_state)

                if step >= start_steps:
                    action = agent.policy(
                        state=tf_prev_state, noise_object=agent.ou_noise, rng=rng
                    )
                else:
                    action = two_mini_random(num_agents)

                step += 1

                # TESTING **********************************
                # time.sleep(0.5)
                # print(action)

                if continuous:
                    # try:
                    #     len(action)
                    # except:
                    #     action = [action]
                    # state, reward, done, info = env.step(action)
                    raise ValueError("not implemented continuous")
                else:
                    # take the argmax to get correct action format [0, 1] instead of [(0.7,0.3), (0.2,0.8)]
                    state, reward, done, info = env.step(
                        np.argmax(action, axis=1).tolist()
                    )

                true_reward = true_reward.__add__(reward)

                terminal_state = np.array(np.invert(done), dtype=np.float32)

                # Reward modification
                if reward_mod:
                    # invert reward
                    # reward = [-r for r in reward]

                    if all(done):
                        reward_addition = [10] * num_agents
                        reward = [
                            reward[i] + reward_addition[i] for i in range(len(reward))
                        ]

                agent.record((prev_state, action, reward, state, terminal_state))

                if step % learn_step == 0:
                    agent.learn()
                    update_target(
                        agent.target_actor.variables,
                        agent.actor_model.variables,
                        agent.tau,
                    )
                    update_target(
                        agent.target_critic.variables,
                        agent.critic_model.variables,
                        agent.tau,
                    )

                episodic_reward = episodic_reward.__add__(reward)

                prev_state = state

                if all(done):
                    break

            list_reward = episodic_reward
            episodic_reward = np.sum(episodic_reward)
            true_reward = np.sum(true_reward)

            ep_reward_list[trial].append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[trial][-mean_number:])
            avg_reward_list[trial].append(avg_reward)
            true_reward_list[trial].append(true_reward)
            true_avg_reward = np.mean(true_reward_list[trial][-mean_number:])
            true_avg_reward_list[trial].append(true_avg_reward)

            if output:
                if reward_mod:
                    print(
                        "Ep {} * AvgReward {:.2f} * true AvgReward {:.2f} * Reward {:.2f} * True Reward {:.2f} * time {:.2f} * step {}".format(
                            ep,
                            avg_reward,
                            true_avg_reward,
                            episodic_reward,
                            true_reward,
                            (time.time() - before),
                            step,
                        )
                    )
                else:
                    print(
                        "Ep {} * AvgReward {:.2f} * ListRew {} * Reward {:.2f} * time {:.2f} * step {}".format(
                            ep,
                            avg_reward,
                            str(list_reward),
                            episodic_reward,
                            (time.time() - before),
                            step,
                        )
                    )

            # Stop if avg is above 'solved'
            if true_avg_reward >= solved:
                break

        # Save weights
        save_weights(agent, env, weights_directory, trial)

    # Plotting graph
    now = datetime.datetime.now()
    timestamp = "{}.{}.{}.{}.{}.{}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    save_name = "{}_{}_{}".format(env.spec.id, agent.continuous, timestamp)
    for idx, p in enumerate(true_avg_reward_list):
        plt.plot(p, label=str(idx))
    plt.xlabel("Episode")
    plt.ylabel("True Avg. Epsiodic Reward (" + str(mean_number) + ")")
    plt.legend()
    try:
        plt.savefig(plots_directory + save_name + ".png")
    except:
        print("Graph save fail")
    plt.show()

    print("total time:", time.time() - start_time, "s")

    if return_data:
        return agent, ep_reward_list, true_reward_list
