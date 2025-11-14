import random
import numpy as np
from scipy.stats import poisson
from tqdm import tqdm
from functools import lru_cache

class CarRental:
    def __init__(self, seed=1234, lambda_rental_1=3, lambda_rental_2=4, 
                 lambda_return_1=3, lambda_return_2=2, max_cars=20):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Poisson parameters for customer arrivals and returns
        self.lambda_rental_1 = lambda_rental_1  # Expected rental requests at location 1
        self.lambda_rental_2 = lambda_rental_2  # Expected rental requests at location 2
        self.lambda_return_1 = lambda_return_1  # Expected returns at location 1
        self.lambda_return_2 = lambda_return_2  # Expected returns at location 2
        self.max_cars = max_cars
        
        # Cache for transition probabilities to avoid recomputation
        self._transition_cache = {}

    def move_cars(self, state, action):
        # morning time
        r1 = state[0] - action
        r2 = state[1] + action
        if r1 > self.max_cars :
            action += r1 - self.max_cars
        if r2 > self.max_cars :
            action -= r2 - self.max_cars
        if r1 < 0:
            action += r1
        if r2 < 0:
            action -= r2

        reward = -2 * abs(action) # cost of moving cars
        if action > 0:
             reward += 2
        r1 = state[0] - action
        r2 = state[1] + action

        if r1 > 10 or r2 > 10:
            reward -= 4

        next_state = [r1, r2]

        return reward, next_state

    @lru_cache(maxsize=10000)
    def poisson_probability(self, n, lambda_param):
        """
        Calculate P(X = n) for Poisson distribution with parameter lambda
        Using scipy.stats.poisson for better performance
        """
        if n < 0:
            return 0.0
        
        # Use scipy.stats.poisson for better performance
        return poisson.pmf(n, lambda_param)

    
    def get_transition_probabilities(self, state):
        """
        Calculate transition probabilities for all possible next states
        Returns dictionary of (next_state, probability) pairs
        Optimized version with vectorized operations and caching
        """
        # Check cache first
        cache_key = (state[0], state[1])
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]
        
        # First, move cars
        # reward_move, state_after_move = self.move_cars(state, action)
        
        # Calculate probabilities for all possible rental/return combinations
        transition_probs = np.zeros((self.max_cars + 1) * (self.max_cars + 1))
        rewards = np.zeros((self.max_cars + 1) * (self.max_cars + 1))

        # Precompute Poisson probabilities for all possible values
        # Use tighter bounds based on lambda values (3-sigma rule)
        max_rentals = min(20, int(3 * max(self.lambda_rental_1, self.lambda_rental_2) + 1))
        max_returns = min(20, int(3 * max(self.lambda_return_1, self.lambda_return_2) + 1))
        
        # Vectorized Poisson probability calculations
        rental1_range = np.arange(max_rentals + 1)
        rental2_range = np.arange(max_rentals + 1)
        return1_range = np.arange(max_returns + 1)
        return2_range = np.arange(max_returns + 1)
        
        # Calculate all Poisson probabilities at once
        prob_rental1_all = poisson.pmf(rental1_range, self.lambda_rental_1)
        prob_rental2_all = poisson.pmf(rental2_range, self.lambda_rental_2)
        prob_return1_all = poisson.pmf(return1_range, self.lambda_return_1)
        prob_return2_all = poisson.pmf(return2_range, self.lambda_return_2)
        
        # Only consider non-negligible probabilities
        rental1_indices = np.where(prob_rental1_all > 1e-10)[0]
        rental2_indices = np.where(prob_rental2_all > 1e-10)[0]
        return1_indices = np.where(prob_return1_all > 1e-10)[0]
        return2_indices = np.where(prob_return2_all > 1e-10)[0]
        
        for rental1_idx in rental1_indices:
            rental1 = rental1_range[rental1_idx]
            prob_rental1 = prob_rental1_all[rental1_idx]
            
            for rental2_idx in rental2_indices:
                rental2 = rental2_range[rental2_idx]
                prob_rental2 = prob_rental2_all[rental2_idx]
                
                for return1_idx in return1_indices:
                    return1 = return1_range[return1_idx]
                    prob_return1 = prob_return1_all[return1_idx]
                    
                    for return2_idx in return2_indices:
                        return2 = return2_range[return2_idx]
                        prob_return2 = prob_return2_all[return2_idx]
                        
                        # Combined probability (assuming independence)
                        prob = prob_rental1 * prob_rental2 * prob_return1 * prob_return2
                        
                        # Calculate actual rentals (limited by available cars)
                        available1 = min(state[0] + return1, self.max_cars)
                        available2 = min(state[1] + return2, self.max_cars)
                        actual_rental1 = min(rental1, available1)
                        actual_rental2 = min(rental2, available2)
                        
                        # Calculate next state
                        next_r1 = np.clip(available1 - actual_rental1, 0, self.max_cars)
                        next_r2 = np.clip(available2 - actual_rental2, 0, self.max_cars)
                        
                        reward = 10 * (actual_rental1 + actual_rental2)
                        state_idx = next_r1 + next_r2 * (self.max_cars + 1)
                        
                        rewards[state_idx] += reward * prob
                        transition_probs[state_idx] += prob
        
        # Cache the result
        result = (transition_probs, rewards)
        self._transition_cache[cache_key] = result
        
        return result

    def clear_cache(self):
        """Clear the transition probability cache"""
        self._transition_cache.clear()

    def precompute_transitions(self, max_action=5):
        """Precompute all transition probabilities for faster policy evaluation"""
        print("Precomputing transition probabilities...")
        num_states = (self.max_cars + 1) * (self.max_cars + 1)
        
        for i in tqdm(range(num_states), desc="Precomputing"):
            state = (i % (self.max_cars + 1), i // (self.max_cars + 1))
            self.get_transition_probabilities(state)
        
        print(f"Cached {len(self._transition_cache)} transition probability calculations")

    def policy_evaluation(self, value_function, policy, gamma=0.9, tol=1e-3):
        """
        Optimized policy evaluation with vectorized operations
        """
        num_states = (self.max_cars + 1) * (self.max_cars + 1)
        
        while True:
            last_value_function = np.copy(value_function)
            
            # Vectorized policy evaluation
            for i in tqdm(range(num_states), desc="Policy Evaluation"):
                state = (i % (self.max_cars + 1), i // (self.max_cars + 1))
                action = policy[i]
                reward_move, state_after_move = self.move_cars(state, action)
                probs, rewards = self.get_transition_probabilities(state_after_move)
                
                # Vectorized value update
                value_function[i] = np.sum(rewards + probs * gamma * last_value_function) + reward_move

            if np.linalg.norm(last_value_function - value_function, np.inf) < tol:
                break
            # print(f"value_function: {value_function}")
        return value_function

    def policy_improvement(self, value_function, policy, gamma=0.9, tol=1e-3):
        """
        Optimized policy improvement with vectorized operations
        """
        num_states = (self.max_cars + 1) * (self.max_cars + 1)

        old_policy = np.copy(policy)
        
        for i in tqdm(range(num_states), desc="Policy Improvement"):
            state = (i % (self.max_cars + 1), i // (self.max_cars + 1))
            max_action = policy[i]
            max_value = value_function[i]
            for action in range(-5, 5 + 1):
                reward_move, state_after_move = self.move_cars(state, action)
                probs, rewards = self.get_transition_probabilities(state_after_move)
                # Vectorized policy improvement
                value = np.sum(rewards + probs * gamma * value_function) + reward_move
                if value > max_value:
                    max_value = value
                    max_action = action
            policy[i] = max_action
   
        return policy, np.all(policy == old_policy)

    def print_policy_grid(self, policy):
        """
        Print the policy as a 2D grid where each cell shows the action to take
        for that state (location1, location2)
        """
        print("\nPolicy Grid (Action to take for each state):")
        # print("Rows: Location 1 cars, Columns: Location 2 cars")
        # print("Actions: negative = move cars from loc1 to loc2, positive = move cars from loc2 to loc1")
        # print()
        
        # Print column headers
        print("     ", end="")
        for j in range(self.max_cars+1):
            print(f"{j:3d}", end="")
        print()
        
        # Print grid
        for i in range(self.max_cars, -1, -1):
            print(f"{i:3d}: ", end="")
            for j in range(self.max_cars + 1):
                state_idx = i + j * (self.max_cars + 1)
                action = policy[state_idx]
                print(f"{action:3d}", end="")
            print()
        
        # print("\nLegend:")
        # print("-5 to -1: Move cars from Location 1 to Location 2")
        # print(" 0: No action")
        # print(" 1 to 5: Move cars from Location 2 to Location 1")


if __name__ == "__main__":
    import time
    
    env = CarRental()
    
    # Test basic functionality
    state = (5, 7)
    action = 0
    reward, next_state = env.move_cars(state, action)
    state = next_state
    print(f"State: {state}, Action: {action}, Reward: {reward}")

    # Demonstrate transition probabilities with timing
    print(f"\n=== Transition Probabilities (Optimized) ===")
    state = (10, 10)  # Use smaller state for faster demo
    action = 0
    
    start_time = time.time()
    transition_probs, rewards = env.get_transition_probabilities(state)
    end_time = time.time()
    
    print(f"Transition probability calculation time: {end_time - start_time:.4f} seconds")
    
    # Verify probabilities sum to 1
    total_transition_prob = np.sum(transition_probs)
    print(f"Total transition probability: {total_transition_prob:.6f}")
    
    # Test caching performance
    print(f"\n=== Testing Cache Performance ===")
    start_time = time.time()
    transition_probs2, rewards2 = env.get_transition_probabilities(state)
    end_time = time.time()
    print(f"Cached transition probability calculation time: {end_time - start_time:.4f} seconds")
    
    # Test policy evaluation with precomputation
    print(f"\n=== Policy Evaluation (Optimized) ===")
    
    # Option 1: Precompute all transitions first (recommended for multiple policy evaluations)
    # print("Precomputing transitions...")
    # start_time = time.time()
    # env.precompute_transitions(max_action=3)  # Smaller action space for demo
    # precompute_time = time.time() - start_time
    # print(f"Precomputation time: {precompute_time:.4f} seconds")
    
    # Now policy evaluation will be much faster
    value_function = np.zeros(21 * 21)
    policy = np.zeros(21 * 21, dtype=int)
    
    start_time = time.time()
    for i in range(100):
        value_function = env.policy_evaluation(value_function, policy)
        policy, policy_stable = env.policy_improvement(value_function, policy)
        
        env.print_policy_grid(policy)
        if policy_stable:
            print(f"Policy converged in {i} iterations")
            break
    eval_time = time.time() - start_time
    print(f"Policy evaluation time: {eval_time:.4f} seconds")
    
    
    
    
   
