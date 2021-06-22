import numpy as np


class QLearning:

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        bandits = env.action_space.n
        states = env.observation_space.n
        Q_values = np.zeros((states,bandits))
        N = np.zeros((states,bandits))
        rewards = []
        initial_state = env.reset()
        for i in range(steps):
          progress = i/steps
          random_action = self._random_action(Q_values[initial_state,:],env,self._get_epsilon(progress))
          new_state,reward,done,info = env.step(random_action)
          N[initial_state,random_action] = N[initial_state,random_action] + 1
          rewards.append(reward)
          alpha = 1/(N[initial_state,random_action])
          Q_values[initial_state,random_action] = Q_values[initial_state,random_action] * (1 - alpha) + \
            alpha * (reward + self.discount * np.argmax(Q_values[new_state,:]))
          initial_state = new_state
          if done:
            initial_state = env.reset()
        s = np.floor(steps/100)  
        rewards = np.array(rewards).reshape((int(len(rewards)/s), int(s)))
        rewards = np.mean(rewards,axis=1)
        return Q_values, rewards

    def _random_action(self, Q, env, eps):
        if np.random.rand(1) <= eps:
          random_action = np.argmax(Q)
        else:
          random_action = env.action_space.sample()
        return random_action
        

    def predict(self, env, state_action_values):
        states,actions,rewards = [],[],[]
        state = env.reset()
        for i in state_action_values:
          random_action = np.argmax(state_action_values[state,:])
          new_state,reward,done,info = env.step(random_action)
          states.append(new_state)
          actions.append(random_action)
          rewards.append(reward)
          if done:
            break
          state = new_state
        return np.array(states),np.array(actions),np.array(rewards)

    def _get_epsilon(self, progress):
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        result = self.epsilon * (1 - progress)
        return result
