import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers

from networks import *
from helpers import *


class Coop_MADDPG:
    def __init__(
        self,
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
    ):

        self.continuous = continuous
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        # This is used to make sure we only sample from used buffer space
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_agents, num_states))
        if self.continuous:
            self.action_buffer = np.zeros(
                (self.buffer_capacity, num_agents, num_actions)
            )
        else:
            self.action_buffer = np.zeros(
                (self.buffer_capacity, num_agents, disc_actions_num)
            )
        self.reward_buffer = np.zeros((self.buffer_capacity, num_agents, 1))
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, num_agents, num_states)
        )
        self.done_buffer = np.zeros((self.buffer_capacity, num_agents, 1), np.float32)
        self.std_dev = std_dev  # For continuous
        self.epsilon = epsilon  # Epsilon greedy for discrete
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.disc_actions_num = disc_actions_num
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.loss_func = loss_func

        self.clip = clip

        self.ou_noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1), theta=theta
        )

        self.actor_model = get_actor(
            num_states, num_actions, continuous, disc_actions_num
        )
        self.critic_model = get_critic(
            num_states, num_agents, num_actions, continuous, disc_actions_num
        )
        self.target_actor = get_actor(
            num_states, num_actions, continuous, disc_actions_num
        )
        self.target_critic = get_critic(
            num_states, num_agents, num_actions, continuous, disc_actions_num
        )

        self.actor_optimizer = optimizers.Adam(
            learning_rate=actor_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=adam_eps,
            amsgrad=amsgrad,
        )
        self.critic_optimizer = optimizers.Adam(
            learning_rate=critic_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=adam_eps,
            amsgrad=amsgrad,
        )
        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    def record(self, obs_tuple):
        # Reuse the same buffer replacing old entries
        index = self.buffer_counter % self.buffer_capacity

        for agent in range(self.num_agents):
            self.state_buffer[index][agent] = obs_tuple[0][agent]
            self.action_buffer[index][agent] = obs_tuple[1][agent]
            self.reward_buffer[index][agent] = obs_tuple[2][agent]
            self.next_state_buffer[index][agent] = obs_tuple[3][agent]
            self.done_buffer[index][agent] = obs_tuple[4][agent]

        self.buffer_counter += 1

    # Calculation of loss and gradients
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        loss_func,
    ):

        # (64, 4, 81) batch, agents, state

        # (64, 4, 81) => (64, 324)
        flat_next_state_last = tf.reshape(
            next_state_batch, (next_state_batch.shape[0], -1)
        )
        print("64, 324", flat_next_state_last.shape)
        # (64, 4, 81) => (256, 81)
        flat_next_state_first = tf.reshape(
            next_state_batch, (-1, next_state_batch.shape[2])
        )
        print("64, 324", flat_next_state_first.shape)
        # (64, 4, 81) => (256, 81)
        flat_state_first = tf.reshape(state_batch, (-1, state_batch.shape[2]))
        print("256, 81", flat_state_first.shape)
        # (64, 4, 81) => (64, 324)
        flat_state_last = tf.reshape(state_batch, (state_batch.shape[0], -1))
        print("64, 324", flat_state_last.shape)
        # (64, 4, 2) => (64, 8)
        flat_action = tf.reshape(action_batch, (action_batch.shape[0], -1))
        print("64, 8", flat_action.shape)

        # calculate per agent loss
        with tf.GradientTape() as tape1:
            # get target actions in shape (256, 2), but we want (64, 8)
            target_actions = self.target_actor(flat_next_state_first, training=True)
            # reshape to get batch size in first axis
            reshaped_target_actions = tf.reshape(target_actions, (self.batch_size, -1))
            # check that the tensor is in the right format
            print(
                "target_actions, should be (64,8) and is", reshaped_target_actions.shape
            )
            # get the target_critic_values and add an dimension (64, 1) to (64, 1, 1)
            target_critic_value = tf.expand_dims(
                self.target_critic(
                    [flat_next_state_last, reshaped_target_actions], training=True
                ),
                -1,
            )
            # output of (64, 4, 1) batch, agent, value
            y = reward_batch + done_batch * self.gamma * target_critic_value
            # reduce to (64, 1)
            y_reduce = tf.math.reduce_sum(y, axis=1)
            print("64, 1", y_reduce.shape)
            # critic value of (64, 1). Input of ([(64, 324), (64, 8)])
            critic_value = self.critic_model(
                [flat_state_last, flat_action], training=True
            )
            print("64, 1", critic_value.shape)
            # final value (64, 1) with same sized inputs
            critic_loss = loss_func(y_reduce, critic_value)  # * (1/self.batch_size)
            print("critic_loss", critic_loss)

            # get the gradient for the critic
            critic_grad = tape1.gradient(
                critic_loss, self.critic_model.trainable_variables
            )

            # Gradient clipping
            critic_clipped_grad, _ = tf.clip_by_global_norm(critic_grad, self.clip)

            # apply gradient
            self.critic_optimizer.apply_gradients(
                zip(critic_clipped_grad, self.critic_model.trainable_variables)
            )

        # calculate per agent
        with tf.GradientTape() as tape2:
            # get actions per all agents (256, 2)
            actions = self.actor_model(flat_state_first, training=True)
            # reshape to (64, 8) to be able to input in critic
            reshaped_actions = tf.reshape(
                actions, (self.batch_size, self.num_agents * self.disc_actions_num)
            )
            # (64, 1) with input (64, 324) and (64, 8)
            critic_value = self.critic_model(
                [flat_state_last, reshaped_actions], training=True
            )
            print("64, 1", critic_value.shape)
            # idk shape
            actor_loss = -tf.math.reduce_mean(critic_value)

            # get the gradient for the critic
            actor_grad = tape2.gradient(
                actor_loss, self.actor_model.trainable_variables
            )

            # Gradient clipping
            actor_capped_grad, _ = tf.clip_by_global_norm(actor_grad, self.clip)

            # apply gradient
            self.actor_optimizer.apply_gradients(
                zip(actor_capped_grad, self.actor_model.trainable_variables)
            )

    def learn(self):
        # Sample only valid data
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])

        self.update(
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            self.loss_func,
        )

    def policy(
        self, state, noise_object=0, use_noise=True, rng=np.random.default_rng()
    ):
        if use_noise:
            if self.continuous:
                # sampled_actions = tf.squeeze(self.actor_model(state))
                # noise = noise_object()
                # sampled_actions = sampled_actions.numpy() + noise
                # # We make sure action is within bounds
                # legal_action = np.clip(sampled_actions, -1, 1)
                # return [np.squeeze(legal_action)][0]
                raise ValueError("not implemented continuous")
            else:
                if (
                    np.random.uniform(0, 1, 1)[0] < self.epsilon
                ):  # not currently using seed
                    # Return random array of actions (can be above sum of 1, but should not matter that much?)
                    return random_action(
                        agents=self.num_agents, disc_actions_num=self.disc_actions_num
                    )
                else:
                    # reshape input from (1, 4, 81) to (1, 324) and output (4,2)
                    return self.actor_model(
                        tf.reshape(state, (state.shape[0], -1))
                    ).numpy()
        else:
            if self.continuous:
                # sampled_actions = tf.squeeze(self.actor_model(state)).numpy()
                # legal_action = np.clip(sampled_actions, -1, 1)
                # return [np.squeeze(legal_action)][0]
                raise ValueError("not implemented continuous")
            else:
                return self.actor_model(tf.reshape(state, (state.shape[0], -1))).numpy()
