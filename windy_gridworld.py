import numpy as np
import random

from tqdm import tqdm


class WindyGridworld:
    def __init__(self, start=np.array([3, 0]), goal=np.array([3, 7]), seed=43):
        self.width = 10
        self.height = 7
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.start = start
        self.goal = goal
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.action_space = {0: np.array([0, 1]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([-1, 0]), 4: np.array([1, 1]), 5: np.array([1, -1]), 6: np.array([-1, 1]), 7: np.array([-1, -1]), 8: np.array([0, 0])}
        self.num_actions = len(self.action_space)
        self.current_state = self.start


    def reset(self):
        self.current_state = self.start
        return self.start

    def step(self, action: int):
        # print(self.action_space[action])
        next_state = self.current_state + self.action_space[action]
        next_state[0] -= self.wind[self.current_state[1]]
        next_state = np.clip(next_state, (0, 0), (self.height - 1, self.width - 1))
        
        self.current_state = next_state
        if np.array_equal(next_state, self.goal):
            return next_state, 0, True
        else:
            return next_state, -1, False


class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.height, env.width, env.num_actions))

    def sample_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(list(self.env.action_space.keys()))
        else:
            return np.argmax(self.Q[state[0], state[1], :])

    def sample_trajectory(self, state, epsilon):
        trajectory = []
        while True:
            action = self.sample_action(state, epsilon)
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, action))
            state = next_state
            if done:
                trajectory.append((state, None))
                return trajectory
        
    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            while True:
                action = self.sample_action(state, self.epsilon)
                next_state, reward, done = self.env.step(action)
                next_action = self.sample_action(next_state, self.epsilon)
                self.Q[state[0], state[1], action] += self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], next_action] - self.Q[state[0], state[1], action])
                state = next_state
                if done:
                    break



if __name__ == "__main__":
    env = WindyGridworld()
    agent = SarsaAgent(env)
    agent.train(3000)
    print(np.max(agent.Q, axis=-1))
    trajectory = agent.sample_trajectory(env.reset(), 0)
    print("trajectory length: ", len(trajectory))
    print(trajectory)
    # for state, action in trajectory:
    #     print(f"{state} -> {action} ->")

    # print(env.reset())
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(0))
    # print(env.step(1))
    # print(env.step(1))
    # print(env.step(1))
    # print(env.step(1))
    # print(env.step(2))
    # print(env.step(2))