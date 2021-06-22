import numpy as np


class MultiArmedBandit:

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000):
        bandits = env.action_space.n
        states = env.observation_space.n
        Q_values = np.zeros((1,bandits))
        rewards = []
        N = np.zeros((bandits))
        env.reset()
        for i in range(steps):
          random_action = self.random_action(Q_values,env,self.epsilon)
          new_state,reward,done,info = env.step(random_action)
          env.render()
          if done:
            env.reset()
          rewards.append(reward)
          N[random_action] = N[random_action] + 1
          Q_values[:,random_action] = Q_values[:,random_action] + (1/N[random_action]) * (reward - Q_values[:,random_action])
        state_action_values = Q_values.copy()
        for i in range(states-1):
          state_action_values = np.concatenate([state_action_values,Q_values],axis=0)
        s = np.floor(steps/100)  
        rewards = np.array(rewards).reshape((int(len(rewards)/s), int(s)))
        rewards = np.mean(rewards,axis=1)
        return state_action_values, rewards

    def random_action(self, Q, env, eps):
        if np.random.rand(1) >= 1 - eps:
          random_action = np.argmax(Q)
        else:
          random_action = env.action_space.sample()
        return random_action

    def predict(self, env, state_action_values):
        states,actions,rewards = [],[],[]
        env.reset()
        for i in state_action_values:
          random_action = np.argmax(i)
          new_state,reward,done,info = env.step(random_action)
          states.append(new_state)
          actions.append(random_action)
          rewards.append(reward)
          if done:
            break
        return np.array(states),np.array(actions),np.array(rewards)
