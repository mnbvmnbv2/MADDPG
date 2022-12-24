import time

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def test(env, actor_weights, continuous, total_episodes=10, render=False, timing=False):
    rewards = []

    for _ in range(total_episodes):
        ep_reward = 0

        before = time.time()

        prev_state = env.reset()
        agent = Coop_MADDPG()
        agent.actor_model.load_weights(actor_weights)

        while True:
            if render:
                env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = agent.policy(state=tf_prev_state, use_noise=False)

            if continuous:
                # try:
                #     len(action)
                # except:
                #     action = [action]
                state, reward, done, _ = env.step(action)
            else:
                state, reward, done, _ = env.step(np.argmax(action))

            ep_reward += reward

            prev_state = state

            if all(done):
                break

        if timing:
            print(str(time.time() - before) + "s")
            rewards.append(ep_reward)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("True reward")
    plt.show()


def rando(env, total_episodes=10, render=False, timing=False, testing=False):
    rewards = []
    for _ in range(total_episodes):
        ep_reward = np.zeros(4)

        before = time.time()

        _ = env.reset()

        while True:
            if render:
                env.render()
            action = env.action_space.sample()  # np.random.randint(0,2,(4)).tolist()
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            # For testing:
            if testing:
                time.sleep(0.5)
                print(done)

            if all(done):
                break

        if timing:
            print(str(time.time() - before) + "s")

        rewards.append(np.sum(ep_reward))

    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("True reward")
    # plt.show()
    return rewards
