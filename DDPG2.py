from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks 

racer = tracks.Racer()

#################################################
# DDPG implementation with parameter space noise# 
#################################################

#############################
###### HYPERPARAMETERS ######
#############################

total_iterations = 50000
# Discount factor
gamma = 0.99
# Target network parameter update factor, for double DQN
tau = 0.005
# Learning rate for actor-critic models
critic_lr = 0.001
aux_lr = 0.001

param_noise_stddev = 0.2

num_states = 5 #we reduce the state dim through observation (see below)
num_actions = 2 #acceleration and steering
print("State Space dim: {}, Action Space dim: {}".format(num_states,num_actions))

upper_bound = 1
lower_bound = -1
print("Min and Max Value of Action: {}".format(lower_bound,upper_bound))

buffer_dim = 50000
batch_size = 64


is_training = True

#pesi

load_weights = False
save_weights = True #beware when saving weights to not overwrite previous data

weights_file_actor = "weights/ddpg2_actor_model_car"
weights_file_critic = "weights/ddpg2_critic_model_car"

#The actor choose the move, given the state
def get_actor():

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, name="perturbable1", activation="relu")(inputs)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(64, name="perturbable2", activation="relu")(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(num_actions, name="perturbable3", activation="tanh")(out)

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
    add_action_noise = False
    # we use the actor modified with parametric noise to generate the action
    sampled_action = tf.squeeze(param_noise_actor(state))
    if add_action_noise:
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

# Updates weigths of the perturbed actor inserting noise too
def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    updates = []
    for var, perturbed_var in zip(actor.trainable_variables, perturbed_actor.trainable_variables):
        if "perturbable" in var.name:
            updates.append(perturbed_var.assign( var + tf.random.normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
    return tf.group(*updates)

# Calculates distance between perturbed actor and clean actor and uses it to update standard deviation
def adapt_param_noise(actor, adaptive_actor, states,current_stddev):
    adoption_coefficient=1.01
    # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
    perturb_adaptive_policy_ops = get_perturbed_actor_updates(actor, adaptive_actor, current_stddev)
    distance = tf.sqrt(tf.reduce_mean(tf.square(actor(states) - adaptive_actor(states))))
    
    if distance > current_stddev:
        # Decrease stddev.
        current_stddev /= adoption_coefficient
    else:
        # Increase stddev.
        current_stddev *= adoption_coefficient
    return current_stddev

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

## TRAINING ##
if load_weights:
    critic_model_temp = keras.models.load_model(weights_file_critic)
    actor_model_temp = keras.models.load_model(weights_file_actor)

# Making the weights equal initially
target_actor_weights = actor_model.get_weights()
target_critic_weights = critic_model.get_weights()
target_actor.set_weights(target_actor_weights)
target_critic.set_weights(target_critic_weights)

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

# Configure perturbed actor.
param_noise_actor = get_actor()
perturb_policy_ops = get_perturbed_actor_updates(actor_model, param_noise_actor, param_noise_stddev)

# Configure separate copy for stddev adoption.
adaptive_param_noise_actor = get_actor()
perturb_adaptive_policy_ops = get_perturbed_actor_updates(actor_model, adaptive_param_noise_actor, param_noise_stddev)

critic_model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(critic_lr))
aux_model.compile(optimizer=tf.keras.optimizers.Adam(aux_lr))


buffer = Buffer(buffer_dim, batch_size)

# History of rewards per episode
ep_reward_list = []
# Average reward history of last few episodes
avg_reward_list = []


# We introduce a probability of doing n empty actions to separate the environment time-step from the agent   
def step(action):
    n = 1
    t = np.random.randint(0,n)
    state ,reward,done = racer.step(action)
    for i in range(t):
        if not done:
            state ,t_r, done =racer.step([0, 0])
            #state ,t_r, done =racer.step(action)
            reward+=t_r
    return (state, reward, done)


def train(total_iterations=total_iterations):
    i = 0
    mean_speed = 0
    current_stddev = param_noise_stddev
    ep = 0
    avg_reward = 0
    while i<total_iterations:
        prev_state = racer.reset()
        episodic_reward = 0
        mean_speed += prev_state[4]
        done = False
        while not(done):
            i = i+1
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            #our policy is always noisy
            action = policy(tf_prev_state)[0]
            # Get state and reward from the environment
            state, reward, done = step(action)
            
            #we distinguish between termination with failure (state = None) and succesfull termination on track completion
            #succesfull termination is stored as a normal tuple
            fail = done and len(state)<5 
            buffer.record((prev_state, action, reward, fail, state))
            if not(done):
                mean_speed += state[4]
        
            episodic_reward += reward

            if buffer.buffer_counter>batch_size:
                states,actions,rewards,dones,newstates= buffer.sample_batch()
                targetQ = rewards + (1-dones)*gamma*(target_critic([newstates,target_actor(newstates)]))
                loss1 = critic_model.train_on_batch([states,actions],targetQ)
                loss2 = aux_model.train_on_batch(states)

                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)
                
                # We calculate adapted standard devation at every step
                current_stddev = adapt_param_noise(actor_model, adaptive_param_noise_actor, states,current_stddev)
            prev_state = state
            
            if i%100 == 0:
                avg_reward_list.append(avg_reward)
        
        # We update the perturbed actor parametric noise only after an episode
        perturb_policy_ops = get_perturbed_actor_updates(actor_model, param_noise_actor, current_stddev)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode {}: Iterations {}, Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep, i, avg_reward,episodic_reward,mean_speed/i))
        print("\n")
        
        if ep>0 and ep%40 == 0:
            print("## Evaluating policy ##")
            tracks.metrics_run(actor_model, 10)
        ep += 1

    if total_iterations > 0:
        if save_weights:
            critic_model.save(weights_file_critic)
            actor_model.save(weights_file_actor)
        # Plotting Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Training steps x100")
        plt.ylabel("Avg. Episodic Reward")
        plt.ylim(-3.5,7)
        plt.show(block=False)
        plt.pause(0.001)
        print("### DDPG2 Training ended ###")
        print("Trained over {} steps".format(i))

if is_training:
    start_t = datetime.now()
    train()
    end_t = datetime.now()
    print("Time elapsed: {}".format(end_t-start_t))


tracks.newrun([actor_model])

