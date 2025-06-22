from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from constants import MODELS_PATH, LOGS_PATH
from envs.portfolio_env import PortfolioEnv
from dataset.rl_dataset import RLDataset

if __name__ == "__main__":
    # 1. Load and prepare dataset
    dataset = RLDataset(
            price_file="asset_prices.csv",
            macro_file="macro_indicators.csv",
            sentiment_file="daily_sentiments.csv",
            reward_columns=["ibov_t-0", "brl_usd_t-0"],
            feature_mode="prices+macros",
            window_size=3,
            rolling_windows=[3, 7],
            normalize=True,
            reward_type="log_return",
        )
    x_train, x_val, x_test, y_train, y_val, y_test = dataset.split(train_end="2015-01-01", val_end="2017-01-01")

    # 2. Create environments
    train_env = PortfolioEnv(x_train, y_train)
    eval_env = PortfolioEnv(x_val, y_val)

    # 3. Train PPO agent
    model = PPO("MlpPolicy", train_env, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_PATH,
        log_path=LOGS_PATH,
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=10_000, callback=eval_callback)

    # 4. Save model
    model.save(f"{MODELS_PATH}/ppo_portfolio")

    obs = eval_env.reset()
    done = False
    returns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        returns.append(reward)

    print(f"Total return: {sum(returns):.4f}")
