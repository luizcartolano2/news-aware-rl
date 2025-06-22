""" PPO Training Script for Portfolio Management.

"""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.portfolio_env import PortfolioEnv
from dataset.rl_dataset import RLDataset
from constants import MODELS_PATH, LOGS_PATH


def make_env(x, y, monitor=True):
    """
    Create a portfolio management environment for training or evaluation.
    This function initializes the `PortfolioEnv` with specified parameters
    and wraps it with a monitor if required.
    :param x: asset features (e.g., prices, indicators)
    :param y: asset rewards (e.g., returns)
    :param monitor: a boolean indicating whether to monitor the environment
    :return: a wrapped `PortfolioEnv` instance
    """
    env = PortfolioEnv(
        x,
        y,
        asset_names=["ibov", "usdbrl"],
        transaction_cost=0.001,  # Lower cost for training
        max_position=0.4,  # More flexible allocations
        reward_scale=10000  # Scaled rewards,
    )
    if monitor:
        env = Monitor(env)
    return env


if __name__ == "__main__":
    # 1. Load and prepare dataset
    dataset = RLDataset(
        price_file="asset_prices.csv",
        macro_file="macro_indicators.csv",
        sentiment_file="daily_sentiments.csv",
        reward_columns=["ibov_t-0", "brl_usd_t-0"],
        feature_mode="all",
        window_size=3,
        rolling_windows=[3, 7],
        normalize=True,
        reward_type="log_return",
    )
    # Split dataset into training, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = dataset.split(
        train_end="2015-01-01",
        val_end="2017-01-01"
    )

    # 2. Create environments
    train_env = DummyVecEnv([lambda: make_env(x_train, y_train)])
    eval_env = DummyVecEnv([lambda: make_env(x_val, y_val)])

    # 3. Train PPO agent
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_PATH,
        log_path=LOGS_PATH,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=10_000, callback=eval_callback)

    # 4. Save and evaluate
    model.save(f"{MODELS_PATH}/ppo_portfolio")

    # Create a separate env for evaluation (no vectorization)
    test_env = make_env(x_test, y_test, monitor=False)

    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    # Get metrics
    metrics = test_env.get_metrics()
    print("\n=== Final Metrics ===")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Visualize allocations
    test_env.render(mode='human')
