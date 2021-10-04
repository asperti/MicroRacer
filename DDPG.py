import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks

racer = tracks.Racer()

########################################

num_states = 5 #we reduce the state dim through observation (see below)
num_actions = 2 #acceleration and steering
print("State Space dim: {}, Action Space dim: {}".format(num_states,num_actions))

upper_bound = 1
lower_bound = -1

print("Min and Max Value of Action: {}".format(lower_bound,upper_bound))

#The actor choose the move, given the state
def get_actor():
    #no special initialization is required
    # Initialize weights between -3e-3 and 3-e3
    #last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
    #outputs = layers.Dense(num_actions, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=last_init)(out)
    #outputs = layers.Activation('tanh')(outputs)
    #outputs = layers.Dense(num_actions, name="out", activation="tanh", kernel_initializer=last_init)(out)
    outputs = layers.Dense(num_actions, name="out", activation="tanh")(out)

    #outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model

def get_actor(train_acceleration=True,train_direction=True):
    # the actor has separate towers for action and speed
    # in this way we can train them separately

    inputs = layers.Input(shape=(num_states,))
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(inputs)
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
    out1 = layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)

    out2 = layers.Dense(32, activation="relu", trainable=train_direction)(inputs)
    out2 = layers.Dense(32, activation="relu",trainable=train_direction)(out2)
    out2 = layers.Dense(1, activation='tanh',trainable=train_direction)(out2)

    outputs = layers.concatenate([out1,out2])

    #outputs = outputs * upper_bound #resize the range, if required
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model

#the critic compute the q-value, given the state and the action
def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out) #Outputs single value

    model = tf.keras.Model([state_input, action_input], outputs, name="critic")

    return model

#Replay buffer
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Max Number of tuples that can be stored
        self.buffer_capacity = buffer_capacity
        # Num of tuples used for training
        self.batch_size = batch_size

        # Current number of tuples in buffer
        self.buffer_counter = 0

        # We have a different array for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Stores a transition (s,a,r,s') in the buffer
    def record(self, obs_tuple):
        s,a,r,T,sn = obs_tuple
        # restart form zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = tf.squeeze(s)
        self.action_buffer[index] = a
        self.reward_buffer[index] = r
        self.done_buffer[index] = T
        self.next_state_buffer[index] = tf.squeeze(sn)

        self.buffer_counter += 1

    def sample_batch(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        s = self.state_buffer[batch_indices]
        a = self.action_buffer[batch_indices]
        r = self.reward_buffer[batch_indices]
        T = self.done_buffer[batch_indices]
        sn = self.next_state_buffer[batch_indices]
        return ((s,a,r,T,sn))

# Slowly updating target parameters according to the tau rate <<1
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def update_weights(target_weights, weights, tau):
    return(target_weights * (1- tau) +  weights * tau)

def policy(state,verbose=False):
    #the policy used for training just add noise to the action
    #the amount of noise is kept constant during training
    sampled_action = tf.squeeze(actor_model(state))
    noise = np.random.normal(scale=0.1,size=2)
    #we may change the amount of noise for actions during training
    noise[0] *= 2
    noise[1] *= .5
    # Adding noise to action
    sampled_action = sampled_action.numpy()
    sampled_action += noise
    #in verbose mode, we may print information about selected actions
    if verbose and sampled_action[0] < 0:
        print("decelerating")

    #Finally, we ensure actions are within bounds
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

#creating models
actor_model = get_actor()
critic_model = get_critic()
#actor_model.summary()
#critic_model.summary()

#we create the target model for double learning (to prevent a moving target phenomenon)
target_actor = get_actor()
target_critic = get_critic()
target_actor.trainable = False
target_critic.trainable = False

#We compose actor and critic in a single model.
#The actor is trained by maximizing the future expected reward, estimated
#by the critic. The critic should be freezed while training the actor.
#For simplicitly, we just use the target critic, that is not trainable.

def compose(actor,critic):
    state_input = layers.Input(shape=(num_states))
    a = actor(state_input)
    q = critic([state_input,a])
    #reg_weights = actor.get_layer('out').get_weights()[0]
    #print(tf.reduce_sum(0.01 * tf.square(reg_weights)))

    m = tf.keras.Model(state_input, q)
    #the loss function of the compound model is just the opposite of the critic output
    m.add_loss(-q)
    return(m)

aux_model = compose(actor_model,target_critic)

## TRAINING ##
#pesi
# ddpg_critic_weigths_32_car0_split.h5 #versione con reti distinte per le mosse. Muove bene ma lento
# ddpg_critic_weigths_32_car1_split.h5 #usual problem: sembra ok

load_weights = True
save_weights = False #beware when saving weights to not overwrite previous data

if load_weights:
    critic_model.load_weights("weights/ddpg_critic_weigths_32_car3_split.h5")
    actor_model.load_weights("weights/ddpg_actor_weigths_32_car3_split.h5")

# Making the weights equal initially
target_actor_weights = actor_model.get_weights()
target_critic_weights = critic_model.get_weights()
target_actor.set_weights(target_actor_weights)
target_critic.set_weights(target_critic_weights)

# Learning rate for actor-critic models
critic_lr = 0.001
aux_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
aux_optimizer = tf.keras.optimizers.Adam(aux_lr)

critic_model.compile(loss='mse',optimizer=critic_optimizer)
aux_model.compile(optimizer=aux_optimizer)

total_episodes = 10
# Discount factor
gamma = 0.99
# Target network parameter update factor, for double DQN
tau = 0.005

buffer = Buffer(50000, 64)

# History of rewards per episode
ep_reward_list = []
# Average reward history of last few episodes
avg_reward_list = []

# custom observation of the state
# it must return an array to be passed as input to both actor and critic

# we extract from the lidar signal the angle dir corresponding to maximal distance max_dir from track borders
# as well as the the distance at adjacent positions.

def max_lidar(observation,angle=np.pi/3,pins=19):
    arg = np.argmax(observation)
    dir = -angle / 2 + arg * (angle / (pins - 1))
    dist = observation[arg]
    if arg == 0:
        distl = dist
    else:
        distl = observation[arg-1]
    if arg == pins-1:
        distr = dist
    else:
        distr = observation[arg+1]
    return(dir,(distl,dist,distr))

def observe(racer_state):
    if racer_state == None:
        return np.array([0]) #not used; we could return None
    else:
        lidar_signal, v = racer_state
        dir, (distl,dist,distr) = max_lidar(lidar_signal)
        return np.array([dir, distl, dist, distr, v])

def train(total_episodes=total_episodes):
    i = 0
    mean_speed = 0

    for ep in range(total_episodes):

        prev_state = observe(racer.reset())
        episodic_reward = 0
        mean_speed += prev_state[4]
        done = False

        while not(done):
            i = i+1

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            #our policy is always noisy
            action = policy(tf_prev_state)[0]
            # Get state and reward from the environment
            state, reward, done = racer.step(action)
            #we distinguish between termination with failure (state = None) and succesfull termination on track completion
            #succesfull termination is stored as a normal tuple
            fail = done and state==None
            state = observe(state)
            buffer.record((prev_state, action, reward, fail, state))
            if not(done):
                mean_speed += state[4]

            buffer.record((prev_state, action, reward, done, state))
            episodic_reward += reward

            states,actions,rewards,dones,newstates= buffer.sample_batch()
            targetQ = rewards + (1-dones)*gamma*(target_critic([newstates,target_actor(newstates)]))

            loss1 = critic_model.train_on_batch([states,actions],targetQ)
            loss2 = aux_model.train_on_batch(states)

            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode {}: Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep, avg_reward,episodic_reward,mean_speed/i))


        avg_reward_list.append(avg_reward)

    if total_episodes > 0:
        if save_weights:
            critic_model.save_weights("weights/ddpg_critic_weigths_32_car3_split.h5")
            actor_model.save_weights("weights/ddpg_actor_weigths_32_car3_split.h5")
        # Plotting Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.show()

train()

def actor(state):
    print("speed = {}".format(state[1]))
    state = observe(state)
    state = tf.expand_dims(state, 0)
    action = actor_model(state)
    print("acc = ",action[0,0].numpy())
    return(action[0])

tracks.newrun(racer,actor)