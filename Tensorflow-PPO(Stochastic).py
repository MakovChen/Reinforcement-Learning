"""
Within 5k episodes, 2/4 attempts solved LunarLanderContinous-v2 (achieved 200 moving averaged rewards).

Some findings are:

It seems better to stop clearing the replay buffer at every episode
It is slow to learn! (perhaps due to some parameters settings...)
Learning rate is important - as we all know :-) (these results haven't fully optimized it yet)

Main parameters are:
γ\gammaγ: 0.99
Replay buffer size: 30k
LR for policy net: 2e-5
LR for value net: 2e-4
Batch size: 128
Number of epochs: 32
Number of hidden layers*: 3
Number of hidden units*: 64
Activation function:
relu for policy net (actor)
tanh for value net (critic)
* for both value and policy net

[reference: https://wandb.ai/takuyamagata/RLRG_AC/reports/Proximal-Policy-Optimization-PPO-on-OpenAI-Gym--VmlldzoxMTY2ODU]
"""

import numpy as np
import tensorflow as tf #version1.13.1
import gym

tf.disable_eager_execution()

class ExperienceReplay():
    """
    // Used to store Actor's experience when interacting with the Gym environment.
    
    【attributes】
    1. __experience: The memory of interacting experience.
    
    【method】
    1. __init__: create a memory when this object has been build.
    2. get_experiments: Return the cumulative experience of state, reward, action, dist, next_state and done sequentially.
    3. addExperience: Add a sample of state, reward, action, dist, next_state and done to __experience. 
       //* definition of variables
       - state: list, [S1, S2...,Sm] (Information about the current environment)
       - reward: float, [R] (The points obtained for the actor's actions in the current state)
       - action: list, [A1, A2...,An] (Control command sent by actors to carriers)
       - dist: dictionary {'mu': [A1, A2...], 'sigma': [A1, A2...]}, (Action's measurement of stochastic distribution,contain mu and sigma.)
       - next_state: list, [S1, S2...,Sm] (New information about the environment after transition)
       - done: boolean (Showing the episode finished or not.)
    4. clearExperience: Remove all experiences in __experience.
  
    """
    def __init__(self):
        self.__experience = {'state':[], 'action':[],'dist':[], 'reward':[], 'next_state':[],'done':[]}

    def get_experiments(self):
        return self.__experience['state'], self.__experience['reward'], self.__experience['action'], self.__experience['dist'], self.__experience['next_state'], self.__experience['done']
        
    def addExperience(self, state = None, reward = None, action = None, dist = None, next_state = None, done = None):
        self.__experience['state'].append(state)
        self.__experience['reward'].append(reward)
        self.__experience['action'].append(action)
        self.__experience['dist'].append(dist)
        self.__experience['next_state'].append(next_state)
        self.__experience['done'].append(done)
        
    def clearExperience(self):
        for col in self.__experience:
            self.__experience[col] = []

class Actor:
    """
    // Used to build and access the actor's neural network
    
    【attributes】
    * All the attributes defined the components of Acotor's network.
    
    【method】
    1. __init__: defined the input, output and the gradients.
    //* definition of variables
       - action_space: dictionary, {'low': [A1bound, A2b...,Anb], 'high': [A1b, A2b...,Anb]} (The Upperbound and Lowerbound for each Action)
       - FLAGS: object, (A variable for store A2C network hyperparameters)
    2. _policy_estimator: defined the network Architecture and the flow of tensors.
    
    """
    def __init__(self, action_space, FLAGS):
        self.FLAGS = FLAGS
        
        #take action
        self.state = tf.placeholder(tf.float32, [None, self.FLAGS.state_size], "state") #input Interface
        self.mu, self.sigma = self._policy_estimator() #forward propagation to get measurements
        self.prob_dist = tf.contrib.distributions.Normal(self.mu, self.sigma) #build a stochastic distribution
        self.action = tf.clip_by_value(self.prob_dist._sample_n(1), action_space['low'], action_space['high']) #sampling a action from distribution and clip values.
        
        #compute surrogate loss functions (the core of Proximal Policy Optimization)
        self.old_mu, self.old_sigma = tf.placeholder(tf.float32, [None, self.FLAGS.action_size], 'old_mu'), tf.placeholder(tf.float32, [None, self.FLAGS.action_size], 'old_sigma')
        self.old_prob_dist = tf.contrib.distributions.Normal(self.old_mu, self.old_sigma)
        self.old_action = tf.placeholder(tf.float32, [None, self.FLAGS.action_size], 'old_action')
        self.ratio = self.prob_dist.prob(self.old_action) / (self.old_prob_dist.prob(self.old_action) + 1e-5)
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.surrogate = self.ratio * self.advantage
        self.loss = -tf.minimum(self.surrogate ,tf.clip_by_value(self.ratio, 1 - self.FLAGS.epsilon, 1 + self.FLAGS.epsilon) * self.advantage)
        
        #the driver of backward propagation
        self.train_op = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Actor).minimize(self.loss)
        
    def _policy_estimator(self):
        #setting the initialize weights and the amount of units for each layer.
        DIM_1, DIM_2, DIM_3 = 64, 64, 64
        inits = [tf.random_normal_initializer(stddev=np.sqrt(1 / self.FLAGS.state_size)),
                 tf.random_normal_initializer(stddev=np.sqrt(1 / DIM_1)),
                 tf.random_normal_initializer(stddev=np.sqrt(1 / DIM_2)),
                 tf.random_normal_initializer(stddev=np.sqrt(1 / DIM_3))]
                 
        #Connecting Network(for generate mu)
        with tf.name_scope('mu'):
            W1, W2, W3, W4 = tf.get_variable('W1_mu', shape=(self.FLAGS.state_size, DIM_1), initializer = inits[0]),
                             tf.get_variable('W2_mu', shape=(DIM_1, DIM_2), initializer = inits[1]),
                             tf.get_variable('W3_mu', shape=(DIM_2, DIM_3), initializer = inits[2]),
                             tf.get_variable('W4_mu', shape=(DIM_3, self.FLAGS.action_size), initializer = inits[3])
            b1, b2, b3 = tf.get_variable('b1_mu', shape=(DIM_1), initializer=inits[0]),
                         tf.get_variable('b2_mu', shape=(DIM_2), initializer=inits[1]),
                         tf.get_variable('b3_mu', shape=(DIM_3), initializer=inits[2])
            h1 = tf.nn.relu(tf.matmul(self.state, W1) + b1)
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
            h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
            mu = tf.squeeze(tf.matmul(h3, W4))
            
        #Connecting Network(for generate sigma)
        with tf.name_scope('sigma'):
            W1, W2, W3, W4 = tf.get_variable('W1_sigma', shape=(self.FLAGS.state_size, DIM_1), initializer = inits[0]),
                             tf.get_variable('W2_sigma', shape=(DIM_1, DIM_2), initializer = inits[1]),
                             tf.get_variable('W3_sigma', shape=(DIM_2, DIM_3), initializer = inits[2]),
                             tf.get_variable('W4_sigma', shape=(DIM_3, self.FLAGS.action_size), initializer = inits[3])
            b1, b2, b3 = tf.get_variable('b1_sigma', shape=(DIM_1), initializer=inits[0]),
                         tf.get_variable('b2_sigma', shape=(DIM_2), initializer=inits[1]),
                         tf.get_variable('b3_sigma', shape=(DIM_3), initializer=inits[2])
            h1 = tf.nn.relu(tf.matmul(self.state, W1) + b1)
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
            h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
            sigma = tf.squeeze(tf.matmul(h3, W4))
            sigma = tf.nn.softplus(sigma) + 1e-5   
        return mu, sigma
    
class Critic:
    """
    // Used to build and access the critic's neural network
    
    【attributes】
    * All the attributes defined the components of Acotor's network.
    
    【method】
    1. __init__: defined the input, output and the gradients.
    //* definition of variables
       - FLAGS: object, (A variable for store A2C network hyperparameters)
    2. _value_estimator: defined the network Architecture and the flow of tensors.
    
    """
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        
        #take state value
        self.state = tf.placeholder(tf.float32, [None, self.FLAGS.state_size], "state") #Input interface
        self.td_target = tf.placeholder(dtype=tf.float32, name="td_target")
        self.state_value = self._value_estimator()
        
        #compute loss functions
        self.loss = tf.reduce_mean(tf.squared_difference(self.td_target, self.state_value))
           
        #the driver of backward propagation
        self.train_op = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Critic).minimize(self.loss)
    
    def _value_estimator(self):
        #setting the initialize weights and the amount of units for each layer.
        DIM_1, DIM_2, DIM_3 = 64, 64, 64
        inits = [tf.random_normal_initializer(stddev=np.sqrt(1 / self.FLAGS.state_size)),
                 tf.random_normal_initializer(stddev=np.sqrt(1 / DIM_1)),
                 tf.random_normal_initializer(stddev=np.sqrt(1 / DIM_2)),
                 tf.random_normal_initializer(stddev=np.sqrt(1 / DIM_3))]
             
        #Connecting Network(for estimate state value)
        with tf.name_scope('state_value'):
            W1, W2, W3, W4 = tf.get_variable('W1_state_value', shape=(self.FLAGS.state_size, DIM_1), initializer = inits[0]),
                             tf.get_variable('W2_state_value', shape=(DIM_1, DIM_2), initializer = inits[1]),
                             tf.get_variable('W3_state_value', shape=(DIM_2, DIM_3), initializer = inits[2]),
                             tf.get_variable('W4_state_value', shape=(DIM_3, self.FLAGS.action_size), initializer = inits[3])
            b1, b2, b3 = tf.get_variable('b1_state_value', shape=(DIM_1), initializer=inits[0]),
                         tf.get_variable('b2_state_value', shape=(DIM_2), initializer=inits[1]),
                         tf.get_variable('b3_state_value', shape=(DIM_3), initializer=inits[2])
            h1 = tf.nn.tanh(tf.matmul(self.state, W1) + b1)
            h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
            h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)
            state_value = tf.squeeze(tf.matmul(h3, W4))
        return state_value
    
class Actor_Critic_Method:
    def __init__(self, session, action_space, FLAGS):
        self.FLAGS = FLAGS
        self.session, self.actor, self.critic = session, Actor(action_space, FLAGS), Critic(FLAGS)
        
    def get_action(self, state):
        state = np.array([state])
        mu = self.session.run(self.actor.mu, feed_dict={self.actor.state: state})
        sigma = self.session.run(self.actor.sigma, feed_dict={self.actor.state: state})
        action = self.session.run(self.actor.action, feed_dict={self.actor.state: state})
        value = self.session.run(self.critic.state_value, { self.critic.state: state})

        return {'mu':mu, 'sigma':sigma}, action, value
    
    def evaluate(self, state, action, reward, dist, epochs): 
        for _ in range(int(epochs * self.FLAGS.importance_sampling_k / self.FLAGS.batch_size )):
            idx = np.random.choice(len(state), size = self.FLAGS.batch_size, replace=False)
            batch_state, batch_reward = np.array([state[i] for i in idx]), np.array([reward[i] for i in idx])
            _, loss = self.session.run([self.critic.train_op, self.critic.loss], { self.critic.state: batch_state , self.critic.td_target: batch_reward})

        for _ in range(int(epochs * self.FLAGS.importance_sampling_k / self.FLAGS.batch_size )):
            idx = np.random.choice(len(state), size = self.FLAGS.batch_size, replace=False)
            batch_state, batch_reward, batch_action = np.array([state[i] for i in idx]), np.array([reward[i] for i in idx]), np.array([action[i] for i in idx])
            state_value = self.session.run(self.critic.state_value, { self.critic.state: batch_state})
            advantage = np.array([batch_reward - state_value]).T
            batch_mu = np.array([dist[i]['mu'] for i in idx])
            batch_sigma = np.array([dist[i]['sigma'] for i in idx])
            _, loss = self.session.run([self.actor.train_op, self.actor.loss], {self.actor.state: batch_state, self.actor.old_action: batch_action, self.actor.old_mu: batch_mu, self.actor.old_sigma: batch_sigma, self.actor.advantage: advantage})

class Proximal_Policy_Optimization:
    def __init__(self, env, FLAGS):
        self.epochs = FLAGS.epochs
        self.FLAGS = FLAGS
        self.env = env
        self.exp_replay = ExperienceReplay()
        self.session = tf.InteractiveSession()
        action_space = {'low': env.action_space.low[0], 'high': env.action_space.high[0]}
        self.policy = Actor_Critic_Method(self.session, action_space, FLAGS)
        self.session.run(tf.global_variables_initializer())
        
    def save_model(self):
        saver = tf.train.Saver().save(self.session, "./model.ckpt")
        
    def update_policy(self, experiments):
        # Monte Carlo Estimation
        Monte_Carlo_reward, discounted_reward, flag = [], 0, False
        for reward, done in zip(reversed(experiments['reward']), reversed(experiments['done'])):
            if done:
                flag = True
                discounted_reward = 0
            discounted_reward = reward + (self.FLAGS.gamma * discounted_reward)
            Monte_Carlo_reward.insert(0, discounted_reward)
        reward, state, action, dist = Monte_Carlo_reward, experiments['state'], experiments['action'], experiments['dist']
        
        #Update Actor-Critics Networks
        self.policy.evaluate(state, action, reward, dist, self.FLAGS.epochs)
        self.exp_replay.clear()

    def playEpisode(self):
        n, iters = 0, 0
        while True:
            state, done, total_reward = self.env.reset(), False, 0
            while done == False:
                self.env.render()
                dist, action, value = self.policy.get_action(state)
                action = action[0]
                next_state, reward, done, _ = self.env.step(action)
                if done == True and reward == -100:
                    reward = 0
                self.exp_replay.addExperience(state = state, reward = reward, action = action, dist = dist, next_state = next_state, done = done)
                total_reward, state = total_reward + reward, next_state
                iters += 1
            print("episodes: %i, MA_total_reward: %.2f" %(n, total_reward))
            if iters > self.FLAGS.importance_sampling_k:
                self.update_policy(self.exp_replay.get_experiments())
                iters = 0
            n += 1
            
env = gym.make('LunarLanderContinuous-v2')

tf.app.flags.DEFINE_float('action_size', env.observation_space.shape[0], '')
tf.app.flags.DEFINE_float('state_size', len(env.observation_space.sample()) , '')
tf.app.flags.DEFINE_float('learning_rate_Actor', 2e-5, 'Learning rate for the policy estimator')
tf.app.flags.DEFINE_float('learning_rate_Critic',  2e-4, 'Learning rate for the state-value estimator')
tf.app.flags.DEFINE_integer('batch_size', 128, 'gradient descent batch size')
tf.app.flags.DEFINE_float('gamma', 0.99, 'Future discount factor')
tf.app.flags.DEFINE_float('epsilon', 0.2, 'clipping surrogate interval')
tf.app.flags.DEFINE_float('epochs', 128, 'PPO allow descenting policy gradient many times with same batch to improve policy network robustly')
tf.app.flags.DEFINE_float('importance_sampling_k', 4000, 'for Monte Carlo heuristic approximation(According CLT, k can setting by everage step to approach target * 30)')

if __name__ == "__main__":
    PPO = Proximal_Policy_Optimization(env, tf.app.flags.FLAGS)
    PPO.playEpisode()
