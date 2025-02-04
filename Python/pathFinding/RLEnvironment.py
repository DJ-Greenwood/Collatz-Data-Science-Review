import numpy as np
import random
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from collections import deque

# Custom Gym Environment for Collatz Pathfinding
class CollatzEnv(gym.Env):
    def __init__(self, start_number):
        super(CollatzEnv, self).__init__()
        self.start_number = start_number
        self.current_number = start_number
        self.steps = 0
        
        # Define action space (0 = n/2, 1 = 3n+1)
        self.action_space = spaces.Discrete(2)
        # Define state space (current number)
        self.observation_space = spaces.Box(low=1, high=10**9, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_number = self.start_number
        self.steps = 0
        return np.array([self.current_number], dtype=np.float32)

    def step(self, action):
        if action == 0 and self.current_number % 2 == 0:
            self.current_number //= 2  # Even step
        elif action == 1:
            self.current_number = 3 * self.current_number + 1  # Odd step

        self.steps += 1

        # Reward function
        if self.current_number == 1:
            reward = 10  # Large reward for reaching 1
            done = True
        else:
            reward = -1  # Penalty for every step taken
            done = False
        
        return np.array([self.current_number], dtype=np.float32), reward, done, {}

    def render(self):
        print(f"Current number: {self.current_number}")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_shape=(self.state_size,), activation="relu"),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Exploit

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Reduce exploration

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights("data/RLModel/collatz_dqn.weights.h5")  # Append `.weights.h5`


def train_dqn(episodes=1000, start_number=987654):
    env = CollatzEnv(start_number)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    batch_size = 32
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):  # Max steps per episode
            action = agent.act(state)  # Choose action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode {e+1}/{episodes}: Steps = {time}, Epsilon = {agent.epsilon:.2f}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    agent.save("data/RLModel/collatz_dqn.h5")
    print("Training complete. Model saved.")

# Train the agent
train_dqn(episodes=10, start_number=31)
