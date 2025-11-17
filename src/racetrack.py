import numpy as np
from dataclasses import dataclass
import random
from scipy.special import softmax
import tqdm

racetrack_string = """
###########
##        G
##        G
##     ####
##     ####
##     ####
##     ####
##     ####
##SSSSS####
"""

@dataclass
class State:
    position: np.ndarray
    velocity: np.ndarray

class Racetrack:
    def __init__(self, racetrack: np.ndarray, starts: set[tuple[int, int]], goals: set[tuple[int, int]], seed: int = 43):
        self.racetrack = racetrack
        self.goals = goals
        self.starts = starts
        self.seed = seed
        self.max_velocity = 5
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.acceleration_map = {}
        count = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                self.acceleration_map[count] = np.array([i, j])
                count += 1

    @classmethod
    def from_string(cls, racetrack_string: str) -> 'Racetrack':
        racetrack = []
        starts = set()
        goals = set()

        for i, line in enumerate(racetrack_string.split('\n')[1:-1]):
            racetrack.append([])
            for j, char in enumerate(line):
                racetrack[-1].append(ord(char))
                if char == 'S':
                    starts.add((i, j))
                elif char == 'G':
                    goals.add((i, j))

        return cls(np.array(racetrack), starts, goals)

    def reset(self) -> State:
        start = random.choice(list(self.starts))
        self.curr_state = State(np.array(start), np.array([0, 0]))
        return self.curr_state

    def step(self, action: int) -> tuple[State, float, bool]:
        acceleration = self.acceleration_map[action]
        velocity = np.clip(self.curr_state.velocity + acceleration, 0, self.max_velocity - 1)
        is_hit, result = self.is_hit(self.curr_state.position, velocity)
        if is_hit and result == "wall":
            return self.reset(), -2, False
        elif is_hit and result == "goal":
            return None, 0, True
        else:
            next_position = np.array([self.curr_state.position[0] - velocity[0], self.curr_state.position[1] + velocity[1]])
            self.curr_state = State(next_position, velocity)
            return self.curr_state, -1, False

    def is_hit(self, position: np.ndarray, velocity: np.ndarray) -> tuple[bool, str]:
        p1 = position
        p2 = np.array([p1[0] - velocity[0], p1[1] + velocity[1]])
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        step = max(abs(dx), abs(dy))

        if step == 0:
            return False, "none"

        step_x = dx / step
        step_y = dy / step

        for i in range(step + 1):
            x = round(p1[0] + step_x * i)
            y = round(p1[1] + step_y * i)
            if self.racetrack[x, y] == ord('#'):
                return True, "wall"
            if self.racetrack[x, y] == ord('G'):
                return True, "goal"
        return False, "none"


class OffPolicyMonteCarloAgent:
    def __init__(self, racetrack: Racetrack, gamma: float = 0.9, epsilon: float = 0.5):
        self.racetrack = racetrack
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = np.zeros((racetrack.racetrack.shape[0], racetrack.racetrack.shape[1], racetrack.max_velocity, racetrack.max_velocity, len(racetrack.acceleration_map))) - 1
        self.C = np.zeros((racetrack.racetrack.shape[0], racetrack.racetrack.shape[1], racetrack.max_velocity, racetrack.max_velocity, len(racetrack.acceleration_map)))

    def sample_trajectory_with_random_policy(self) -> list[tuple[State, int, float]]:
        state = self.racetrack.reset()
        trajectory = []
        while True:
            action = random.choice(range(len(self.racetrack.acceleration_map)))
            next_state, reward, done = self.racetrack.step(action)
            # print(state, action, reward, next_state)
            trajectory.append((state, action, reward))
            if done:
                break
            state = next_state
        return trajectory

    def sample_trajectory_with_epsilon_greedy_policy(self) -> list[tuple[State, int, float, float]]:
        state = self.racetrack.reset()
        trajectory = []
        while True:
            if random.random() < self.epsilon:
                action = random.choice(range(len(self.racetrack.acceleration_map)))
                prob = self.epsilon / len(self.racetrack.acceleration_map)
            else:
                action = np.argmax(self.Q_table[state.position[0], state.position[1], state.velocity[0], state.velocity[1], :])
                prob = 1 - self.epsilon
            next_state, reward, done = self.racetrack.step(action)
            trajectory.append((state, action, reward, prob))
            if done:
                break
            state = next_state
        return trajectory

    def sample_trajectory_with_softmax_policy(self) -> list[tuple[State, int, float, float]]:
        state = self.racetrack.reset()
        trajectory = []
        while True:
            probs = softmax(self.Q_table[state.position[0], state.position[1], state.velocity[0], state.velocity[1], :], axis=None)
            action = np.random.choice(range(len(self.racetrack.acceleration_map)), p=probs)
            next_state, reward, done = self.racetrack.step(action)
            trajectory.append((state, action, reward, probs[action]))
            if done:
                break
            state = next_state
        return trajectory
    
    def sample_trajectory_with_greedy_policy(self) -> list[tuple[State, int, float]]:
        state = self.racetrack.reset()
        trajectory = []
        for _ in range(100):
            action = np.argmax(self.Q_table[state.position[0], state.position[1], state.velocity[0], state.velocity[1], :])
            next_state, reward, done = self.racetrack.step(action)
            trajectory.append((state, action, reward))
            if done:
                break
            state = next_state
        return trajectory

    def train(self, num_episodes: int):
        pos_prob = (1 - 0.05 + 0.05 / 9)
        neg_prob = 0.05 / 9
        for i in tqdm.tqdm(range(num_episodes)):
            self.epsilon = 0.3 - (0.25 * i / num_episodes)
            trajectory = self.sample_trajectory_with_epsilon_greedy_policy()
            G = 0
            W = 1
            for j, (state, action, reward, prob) in enumerate(trajectory[::-1]):
                if j > 100:
                    break
                G = self.gamma * G + reward
                self.C[state.position[0], state.position[1], state.velocity[0], state.velocity[1], action] += W
                self.Q_table[state.position[0], state.position[1], state.velocity[0], state.velocity[1], action] += (W / self.C[state.position[0], state.position[1], state.velocity[0], state.velocity[1], action]) * (G - self.Q_table[state.position[0], state.position[1], state.velocity[0], state.velocity[1], action])
                a = np.argmax(self.Q_table[state.position[0], state.position[1], state.velocity[0], state.velocity[1], :])
                if a == action:
                    W = W * pos_prob / prob
                else:
                    W = W * neg_prob / prob
        

if __name__ == "__main__":
    racetrack = Racetrack.from_string(racetrack_string)
    print(racetrack.racetrack)
    print("starts", racetrack.starts)
    print("goals", racetrack.goals)

    agent = OffPolicyMonteCarloAgent(racetrack)
    agent.train(3000)
    print(agent.sample_trajectory_with_greedy_policy())
    print(agent.sample_trajectory_with_greedy_policy())