import gym
import numpy as np
from gym import spaces
from typing import Tuple


class PortfolioEnv(gym.Env):
    """
    Custom Gym environment for portfolio optimization.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int = 1,
        transaction_cost: float = 0.001
    ):
        super(PortfolioEnv, self).__init__()

        self.x = x
        self.y = y
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.n_assets = y.shape[1]
        self.n_features = x.shape[1]

        self.current_step = 0
        self.done = False
        self.prev_allocation = np.ones(self.n_assets) / self.n_assets

        # Action: portfolio weights for each asset (bounded, continuous)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        # Observation: current feature vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        Reset environment to the initial state.
        """
        self.current_step = 0
        self.done = False
        self.prev_allocation = np.ones(self.n_assets) / self.n_assets
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        return self.x[self.current_step]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step within the environment.
        """
        # Normalize portfolio weights
        action = np.clip(action, 0, 1)
        action = action / np.sum(action + 1e-8)  # prevent divide by 0

        # Get reward: dot product between action and future return
        future_return = self.y[self.current_step]
        reward = float(np.dot(action, future_return))

        # Apply transaction cost penalty (L1 change in portfolio)
        cost = self.transaction_cost * np.sum(np.abs(action - self.prev_allocation))
        reward -= cost

        self.prev_allocation = action

        self.current_step += 1
        if self.current_step >= len(self.x) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        print(f"Step {self.current_step}")

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space
