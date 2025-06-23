""" Portfolio Optimization Environment
This code defines a Gymnasium environment for portfolio optimization, allowing for realistic trading scenarios with
features like transaction costs, position limits, and risk-adjusted rewards. It supports asset name tracking and
provides comprehensive performance metrics.
The environment is designed to be flexible and extensible, suitable for reinforcement learning applications
in financial markets.

"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt


class PortfolioEnv(gym.Env):
    """
    Portfolio Optimization Environment
    This environment simulates a portfolio management scenario where an agent can allocate capital across multiple assets.
    It includes features such as transaction costs, position limits, and risk-adjusted rewards.
    Attributes:
        x (np.ndarray): Feature matrix (shape: [n_samples, n_features])
        y (np.ndarray): Asset returns matrix (shape: [n_samples, n_assets])
        asset_names (List[str]): Names of the assets
        window_size (int): Size of the observation window
        transaction_cost (float): Transaction cost per trade
        max_position (float): Maximum position size allowed
        reward_scale (float): Scaling factor for rewards
        lookback_window (int): Number of steps to consider for risk calculation

    Methods:
        reset(): Resets the environment to the initial state.
        step(action): Executes one time step with the given action.
        render(mode='human'): Renders the current state of the environment.
        get_metrics(): Returns comprehensive performance metrics.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        asset_names: Optional[List[str]] = None,
        window_size: int = 1,
        transaction_cost: float = 0.002,
        max_position: float = 0.3,
        reward_scale: float = 100.0,
        lookback_window: int = 21
    ):
        super().__init__()

        # Validate inputs
        assert len(x) == len(y), "Features and returns must have same length"
        assert max_position > 0, "Max position must be positive"

        self.x = x
        self.y = y
        self.asset_names = asset_names if asset_names else [f"Asset_{i+1}" for i in range(y.shape[1])]
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_scale = reward_scale
        self.lookback_window = lookback_window

        # Environment dimensions
        self.n_assets = y.shape[1]
        self.n_features = x.shape[1]
        self.max_steps = len(x) - 1

        # Initialize state
        self.current_step = window_size - 1
        self.portfolio_value = 1.0
        self.prev_allocation = np.ones(self.n_assets) / self.n_assets

        # Tracking
        self.history = {
            'values': [1.0],
            'allocations': [],
            'market_returns': [],
            'portfolio_returns': [],
            'costs': []
        }

        # Spaces
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "market_data": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_features,),
                dtype=np.float32
            ),
            "allocation": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_assets,),
                dtype=np.float32
            ),
            "portfolio_value": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            )
        })

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to the initial state.
        This method initializes the portfolio value, resets the step counter, and clears the history.
        It also returns the initial observation.
        :param seed: a random seed for reproducibility
        :param options: a dictionary of options for resetting the environment
        :return: a tuple containing the initial observation and an empty info dictionary
        """
        super().reset(seed=seed)

        self.current_step = self.window_size - 1
        self.portfolio_value = 1.0
        self.prev_allocation = np.ones(self.n_assets) / self.n_assets

        self.history = {
            'values': [1.0],
            'allocations': [],
            'market_returns': [],
            'portfolio_returns': [],
            'costs': []
        }

        return self._get_observation(), {}

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation of the environment.
        This method returns the current market data, previous allocation, and portfolio value.
        :return: a dictionary containing the current market data, allocation, and portfolio value
        """
        return {
            "market_data": self.x[self.current_step],
            "allocation": self.prev_allocation,
            "portfolio_value": np.array([self.portfolio_value], dtype=np.float32)
        }

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process the action to ensure it is valid.
        This method clips the action to the maximum position size and normalizes it to sum to 1.
        It ensures that the action is a valid allocation vector.
        :param action: a numpy array representing the action (asset allocations)
        :return: a normalized numpy array representing the processed action
        """
        action = np.clip(action, 0, self.max_position)
        action = action / (np.sum(action) + 1e-8)
        return action.astype(np.float32)

    def _calculate_risk(self, returns: np.ndarray) -> float:
        """
        Calculate the risk of the portfolio based on recent returns.
        This method computes the standard deviation of the returns over the lookback window.
        It returns a risk measure that can be used to adjust rewards.
        :param returns: a numpy array of recent portfolio returns
        :return: a float representing the risk (standard deviation of returns)
        """
        if len(returns) < 2:
            return 1.0
        return np.std(returns)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        This method processes the action, calculates the portfolio return, applies transaction costs,
        updates the portfolio value, and computes the reward. It also tracks various metrics.
        :param action: a numpy array representing the action (asset allocations)
        :return: a tuple containing the next observation, reward, termination status, truncation status, and info dictionary
        """
        if self.current_step >= self.max_steps:
            raise ValueError("Episode has already completed")

        action = self._process_action(action)
        current_returns = self.y[self.current_step]

        # Track metrics
        self.history['market_returns'].append(current_returns)
        portfolio_return = np.dot(action, current_returns)
        self.history['portfolio_returns'].append(portfolio_return)

        # Apply transaction cost
        turnover = np.sum(np.abs(action - self.prev_allocation))
        cost = self.transaction_cost * turnover
        self.history['costs'].append(cost)

        # Update portfolio
        self.portfolio_value *= (1 + portfolio_return) * (1 - cost)
        self.history['values'].append(self.portfolio_value)
        self.history['allocations'].append(action)
        self.prev_allocation = action

        # Calculate reward
        recent_returns = self.history['portfolio_returns'][-self.lookback_window:]
        risk = self._calculate_risk(recent_returns)
        reward = (portfolio_return / (risk + 1e-8)) * self.reward_scale - cost

        # Update step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "cost": cost,
            "sharpe_ratio": reward / self.reward_scale,
            "asset_allocations": dict(zip(self.asset_names, action))
        }

        return self._get_observation(), reward, terminated, False, info

    def render(self, mode: str = 'human', filename: str = '') -> None:
        """
        Render the environment's current state.
        This method visualizes the portfolio value, asset allocations, returns comparison, and transaction costs.
        It supports both human-readable and ANSI text modes.
        :param filename: a string for saving the plot (optional)
        :param mode: 'human' for graphical rendering, 'ansi' for text output
        :return: a NoneType
        """
        if mode == 'human':
            plt.figure(figsize=(14, 8))

            # Portfolio Value
            plt.subplot(2, 2, 1)
            plt.plot(self.history['values'])
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Value ($)')

            # Asset Allocations
            plt.subplot(2, 2, 2)
            allocations = np.array(self.history['allocations'])
            for i in range(allocations.shape[1]):
                plt.plot(allocations[:, i], label=self.asset_names[i])
            plt.title('Asset Allocation History')
            plt.xlabel('Time Step')
            plt.ylabel('Allocation %')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Returns Comparison
            plt.subplot(2, 2, 3)
            plt.plot(self.history['portfolio_returns'], label='Portfolio')
            plt.plot([np.mean(r) for r in self.history['market_returns']],
                    label='Market Average')
            plt.title('Returns Comparison')
            plt.xlabel('Time Step')
            plt.ylabel('Daily Return')
            plt.legend()

            # Transaction Costs
            plt.subplot(2, 2, 4)
            plt.plot(self.history['costs'])
            plt.title('Transaction Costs')
            plt.xlabel('Time Step')
            plt.ylabel('Cost ($)')

            plt.tight_layout()
            if filename == '':
                plt.show()
            else:
                plt.savefig(filename)
        elif mode == 'ansi':
            print(f"\nStep {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print("Current Allocations:")
            for name, alloc in zip(self.asset_names, self.prev_allocation):
                print(f"  {name}: {alloc:.1%}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics for the portfolio.
        This method computes cumulative return, annualized return, volatility, Sharpe ratio,
        maximum drawdown, turnover, and asset allocations.
        :return: a dictionary containing various performance metrics
        """
        returns = np.array(self.history['portfolio_returns'])
        cum_return = self.portfolio_value - 1

        if len(returns) == 0:
            return {
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "asset_allocations": dict(zip(self.asset_names, self.prev_allocation))
            }

        # Calculate metrics
        n_periods = len(returns)
        annualized_return = (1 + cum_return) ** (252/n_periods) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_dd = (np.maximum.accumulate(self.history['values']) - self.history['values']).max()

        # Allocation statistics
        allocs = np.array(self.history['allocations'])
        avg_allocs = np.mean(allocs, axis=0) if len(allocs) > 0 else self.prev_allocation

        return {
            "cumulative_return": float(cum_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "turnover": float(np.sum(np.abs(np.diff(allocs, axis=0)))),
            "average_allocations": dict(zip(self.asset_names, avg_allocs)),
            "final_allocations": dict(zip(self.asset_names, self.prev_allocation))
        }

    def close(self):
        """
        Close the environment and clean up resources.
        :return: a NoneType
        """
        plt.close('all')
