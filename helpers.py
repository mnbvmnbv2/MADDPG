import datetime
import tensorflow as tf
import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
        
def fixed(x, episode):
    return x

def save_weights(agent, env, weights_directory, trial):
    now = datetime.datetime.now()
    timestamp = "{}.{}.{}.{}.{}.{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    save_name = "{}_{}_{}".format(env.spec.id, agent.continuous, timestamp)
    try:
        agent.actor_model.save_weights(weights_directory + 'actor-trial' + str(trial) + '_' + save_name + '.h5')
    except:
        print('actor save fail')
    try:
        agent.critic_model.save_weights(weights_directory + 'critic-trial' + str(trial) + '_' + save_name + '.h5')
    except:
        print('critic save fail')

def two_mini_random(num_agents):
    out = []
    for _ in range(num_agents):
        bat = np.random.uniform(0,1,2)
        bat[1] = 1 - bat[0]
        out.append(bat.tolist())
    return out