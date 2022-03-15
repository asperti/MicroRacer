from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
import tracks

num_states = 5 
num_actions = 2 

upper_bound = 1
lower_bound = -1


# Learning rate for actor-critic models
critic_lr = 3e-4
actor_lr = 3e-4

# Number of episodes
total_iterations = 600
# Mini-batch size for training
batch_size = 64
# Number of training steps with the same episode
epochs = 10

gamma = 0.99
gae_lambda = 0.95
policy_clip = tf.constant(0.25, dtype=tf.float32)

target_entropy = tf.constant(0.01, dtype=tf.float32)

target_kl = 0.01

is_training = True

load_weights = False
save_weights = True

weights_file_actor = "weights/ppo_actor_model_car"
weights_file_critic = "weights/ppo_critic_model_car"

#The actor choose the move, given the state
class Get_actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(64, activation="tanh")
        self.d2 = layers.Dense(64, activation="tanh")
        self.m = layers.Dense(num_actions, activation="tanh")
        
    def call(self, s):
        out = self.d1(s)
        out = self.d2(out)
        mu = self.m(out)
        sigma = 0.2
        return  mu, sigma
    
    @property  
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.m.trainable_variables


#the critic compute the value, given the state 
class Get_critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(64, activation="tanh")
        self.d2 = layers.Dense(64, activation="tanh")
        self.o = layers.Dense(1)
        
    def call(self, inputs):
        out = self.d1(inputs)
        out = self.d2(out)
        q = self.o(out)
        return q
    
    @property
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.o.trainable_variables
    

#trajectories buffer
class Buffer:
    def __init__(self, batch_size):
        self.states=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        self.val=[]
        self.logp=[]
        self.batch_size = batch_size

    def sample_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.rewards),np.array(self.dones), np.array(self.val), np.array(self.logp), batches
            
    def record(self, state, action, reward, done, val, logp):
        self.states.append(tf.squeeze(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.val.append(tf.squeeze(val))
        self.logp.append(tf.squeeze(logp))
       
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear() 
        self.val.clear()
        self.logp.clear()
                   
class Agent:
    def __init__(self):
        self.racer = tracks.Racer()
        self.actor_model = Get_actor()
        self.critic_model = Get_critic()
        self.buffer = Buffer(batch_size)     
        ## TRAINING ##
        self.critic_model(layers.Input(shape=(num_states)))
        self.actor_model(layers.Input(shape=(num_states)))
        if load_weights:
            self.critic_model = keras.models.load_model(weights_file_critic)
            self.actor_model = keras.models.load_model(weights_file_actor)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_model.compile(optimizer=self.actor_optimizer)
        self.critic_model.compile(loss="mse",optimizer=self.critic_optimizer)
        # History of rewards per episode
        self.ep_reward_list = []
        # Average reward history of last few episodes
        self.avg_reward_list = []
        # Keep track of how many training steps has been done
        
    #Returns an action sampled from the normal distribution returned by the actor and it's relative log probability.
    #If an action is passed returns it's log probability.
    def get_action_and_logp(self, states, actions=None):
        mu, sigma = self.actor_model(states)
        dist = tfp.distributions.Normal(mu, sigma)
        if actions is None:
            # Use of the reparameterization trick 
            actions = mu + sigma * tfp.distributions.Normal(0,1).sample(num_actions)   
        log_p = dist.log_prob(actions)
        
        if len(log_p.shape)>1:
            log_p = tf.reduce_sum(log_p,1)
        else:
            log_p = tf.reduce_sum(log_p)
        log_p = tf.expand_dims(log_p, 1)
        
        valid_action  = K.clip(actions, lower_bound, upper_bound)
        
        return  valid_action, log_p
        
        
    def gae(self, values, rewards, masks, lastvalue):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i==len(rewards)-1:
                nextvalue=lastvalue
            else:
                nextvalue=values[i+1]
            delta=rewards[i]+gamma*nextvalue*masks[i]-values[i]  
            gae=delta+gamma*gae_lambda*masks[i]*gae
            returns.insert(0, gae+values[i])
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        return np.array(returns), advantages
    
    def update_networks(self, last_value=0): 
        states, actions, rewards, dones, values, old_logp, batches = self.buffer.sample_batch()
        returns, advantages = self.gae(values, rewards, dones, last_value)     
        # Train using mini-batches
        for batch in batches:
            s_batch = tf.convert_to_tensor(states[batch], dtype=tf.float32)
            a_batch = tf.convert_to_tensor(actions[batch], dtype=tf.float32)
            adv_batch = tf.expand_dims(tf.convert_to_tensor(advantages.numpy()[batch], dtype=tf.float32),1)
            ret_batch =  tf.expand_dims(tf.convert_to_tensor(returns[batch], dtype=tf.float32),1)
            ologp_batch = tf.expand_dims(tf.convert_to_tensor(old_logp[batch], dtype=tf.float32),1)
            for e in range(epochs):
                with tf.GradientTape() as tape:
                    tape.watch(self.actor_model.trainable_variables)
                    _,logp_batch = self.get_action_and_logp(tf.stack(s_batch), tf.stack(a_batch)) 
                    ratio = tf.exp(logp_batch-ologp_batch)
                    weighted_ratio = ratio*adv_batch
                    weighted_clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1- policy_clip, clip_value_max=1+ policy_clip)*adv_batch
                    min_wr = tf.minimum(weighted_ratio, weighted_clipped_ratio)- target_entropy*logp_batch
                    loss = -tf.reduce_mean(min_wr)            
                grad = tape.gradient(loss, self.actor_model.trainable_variables)    
                self.actor_model.optimizer.apply_gradients(zip(grad, self.actor_model.trainable_variables))
                
                c_loss = self.critic_model.train_on_batch(s_batch,ret_batch)
                
                # We use approximatation of Kullback–Leibler divergence to early stop training epochs
                _,logp = self.get_action_and_logp(s_batch, a_batch) 
                kl = tf.reduce_mean(ologp_batch-logp)
                if kl > 1.5*target_kl:
                    print("early stopping - max kl reached at epoch {}".format(e))
                    break

        # We empty the buffer after policy update
        self.buffer.clear()
     
    # We introduce a probability of doing n empty actions to separate the environment time-step from the agent   
    def step(self, action):
        n = 2
        t = np.random.randint(0,n)
        state ,reward,done = self.racer.step(action)
        for i in range(t):
            if not done:
                state ,t_r, done = self.racer.step([0, 0])
                #state ,t_r, done =racer.step(action)
                reward+=t_r
        return (state, reward, done)
          
    def train(self):
        i = 0
        mean_speed = 0
        for ep in range(total_iterations):
            state = self.racer.reset()
            done = False
            episodic_reward = 0
            
            while not done:    
                i+=1
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action, logp = self.get_action_and_logp(state)
                value = self.critic_model(state)
                #action = K.clip(action, lower_bound, upper_bound)
                action = tf.squeeze(action)
                nstate, reward, done = self.step(action)
                self.buffer.record(state, action, reward, not done, value, logp)
                if not done:
                    mean_speed += nstate[4]
                state = nstate
                episodic_reward += reward
              
            # training after a complete episode 
            self.update_networks()
               
            self.ep_reward_list.append(episodic_reward) 
            avg_reward = np.mean(self.ep_reward_list[-40:])
            #avg_reward = np.mean(self.ep_reward_list)
            print("Episode {}: Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep, avg_reward,episodic_reward,mean_speed/i))
            print("\n")
            self.avg_reward_list.append(avg_reward)

        if total_iterations > 0:
            if save_weights:
                self.critic_model.save(weights_file_critic)
                self.actor_model.save(weights_file_actor)
            # Plotting Episodes versus Avg. Rewards
            plt.plot(self.avg_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Episodic Reward")
            plt.ylim(-3.5,7)
            plt.show(block=False)
            plt.pause(0.001)
        print("### PPO Training ended ###")
            

    def launch(self):
        if is_training:
            start_t = datetime.now()
            self.train()
            end_t = datetime.now()
            print("Time elapsed: {}".format(end_t-start_t))
        tracks.newrun([self.actor_model])

        
        
ppo_agent = Agent()
ppo_agent.launch()