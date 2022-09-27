from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import initializers

# Networks
def get_actor(num_states, num_actions, continuous, disc_actions_num):
    
    ### ACTOR NETWORK ###
    
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.LayerNormalization(axis=1)(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.LayerNormalization(axis=1)(out)
    
    if continuous:
        outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=initializers.RandomNormal(stddev=0.03))(out)
    else:
        outputs = layers.Dense(disc_actions_num, activation="softmax", kernel_initializer=initializers.RandomNormal(stddev=0.03))(out)
    
    return Model(inputs, outputs)

def get_critic(num_states, num_agents, num_actions, continuous, disc_actions_num):
    
    ### CRITIC NETWORK ###
    
    state_input = layers.Input(shape=(num_states * num_agents))
    state_out = layers.Dense(64, activation="relu")(state_input)
    
    if continuous:
        action_input = layers.Input(shape=(num_actions * num_agents))
    else:
        action_input = layers.Input(shape=(disc_actions_num * num_agents))
    action_out = layers.Dense(64, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.LayerNormalization(axis=1)(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.LayerNormalization(axis=1)(out)
    outputs = layers.Dense(num_actions)(out)

    return Model([state_input, action_input], outputs)