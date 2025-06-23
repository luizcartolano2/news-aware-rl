from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from constants import MODELS_PATH, LOGS_PATH, IMAGES_PATH, TEXT_PATH
from dataset.rl_dataset import RLDataset
from envs.portfolio_env import PortfolioEnv


class PortfolioTrainer:
    def __init__(self, price_file, macro_file, sentiment_file, reward_columns, models_path=MODELS_PATH, logs_path=LOGS_PATH):
        self.price_file = price_file
        self.macro_file = macro_file
        self.sentiment_file = sentiment_file
        self.reward_columns = reward_columns
        self.asset_names = self.__load_asset_names(reward_columns)
        self.models_path = models_path
        self.logs_path = logs_path
        self.images_path = f"{self.models_path}/images"

    @staticmethod
    def __load_asset_names(reward_columns):
        """
        Extract asset names from reward columns.
        Assumes reward columns are formatted as 'asset_name_t-0'.
        :param reward_columns: List of reward column names.
        :return: List of asset names.
        """
        return [col.split('_')[0] for col in reward_columns]

    def __make_env(self, x_data, y_data, monitor: bool = True):
        """
        Create a portfolio management environment for training or evaluation.
        This function initializes the `PortfolioEnv` with specified parameters
        and wraps it with a monitor if required.
        :param x_data: asset features (e.g., prices, indicators)
        :param y_data: asset rewards (e.g., returns)
        :param monitor: a boolean indicating whether to monitor the environment
        :return: a wrapped `PortfolioEnv` instance
        """
        env = PortfolioEnv(
            x_data, y_data,
            asset_names=self.asset_names,
            transaction_cost=0.001,
            max_position=0.4,
            reward_scale=10000
        )
        return Monitor(env) if monitor else env

    def prepare_data(self, feature_mode, reward_type, train_end, val_end):
        """
        Prepare the dataset for training and validation.
        This function initializes the `RLDataset` with the provided parameters
        and splits it into training, validation, and test sets.
        :param feature_mode: a string indicating the feature mode (e.g., 'all', 'price_only')
        :param reward_type: a string indicating the type of reward (e.g., 'log_return', 'simple_return')
        :param train_end: a string or datetime indicating the end date for the training set
        :param val_end: a string or datetime indicating the end date for the validation set
        :return: a tuple containing the training, validation, and test datasets
        """
        dataset = RLDataset(
            price_file=self.price_file,
            macro_file=self.macro_file,
            sentiment_file=self.sentiment_file,
            reward_columns=self.reward_columns,
            feature_mode=feature_mode,
            window_size=3,
            rolling_windows=[3, 7],
            normalize=True,
            reward_type=reward_type,
        )
        return dataset.split(train_end=train_end, val_end=val_end)

    def train_agent(self, agent_type, train_end, val_end, feature_mode, reward_type="log_return", total_steps=100_000):
        """Train and evaluate a specific configuration"""
        # Prepare data
        x_train, x_val, x_test, y_train, y_val, y_test = self.prepare_data(
            feature_mode=feature_mode,
            reward_type=reward_type,
            train_end=train_end,
            val_end=val_end
        )

        # Create environments
        train_env = DummyVecEnv([lambda: self.__make_env(x_train, y_train)])
        eval_env = DummyVecEnv([lambda: self.__make_env(x_val, y_val)])
        test_env = self.__make_env(x_test, y_test, monitor=False)

        # Initialize model
        if agent_type.lower() == "a2c":
            model = A2C(
                "MultiInputPolicy",
                train_env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                verbose=1,
            )
        elif agent_type.lower() == "ppo":
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
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Train
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_path}/{agent_type}_{feature_mode}_{reward_type}",
            log_path=f"{self.logs_path}/{agent_type}_{feature_mode}_{reward_type}",
            eval_freq=1000,
            deterministic=True,
            render=False,
        )
        model.learn(total_timesteps=total_steps, callback=eval_callback)
        model.save(f"{self.models_path}/{agent_type}_{feature_mode}_{reward_type}_final")

        # Evaluate train dataset
        obs = train_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated = train_env.step(action)
            done = terminated or truncated

        train_metrics = train_env.env_method('get_metrics')[0]
        train_metrics.update({
            'agent_type': agent_type,
            'feature_mode': feature_mode,
            'reward_type': reward_type
        })

        # Evaluate val dataset
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated = eval_env.step(action)
            done = terminated or truncated

        eval_metrics = train_env.env_method('get_metrics')[0]
        eval_metrics.update({
            'agent_type': agent_type,
            'feature_mode': feature_mode,
            'reward_type': reward_type
        })

        # Evaluate test dataset
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated

        test_metrics = test_env.get_metrics()
        test_metrics.update({
            'agent_type': agent_type,
            'feature_mode': feature_mode,
            'reward_type': reward_type
        })
        test_env.render(mode='human', filename=f"{IMAGES_PATH}/{agent_type}_{feature_mode}_{reward_type}_test_results.jpg")

        return train_metrics, eval_metrics, test_metrics

    def run_experiments(self, agent_types, feature_modes, reward_types, train_end, val_end, total_steps=50_000):
        all_metrics = []

        for agent in agent_types:
            for feature_mode in feature_modes:
                for reward_type in reward_types:
                    print(f"\nTraining {agent} with {feature_mode} features and {reward_type} reward")
                    try:
                        train_metrics, eval_metrics, test_metrics = self.train_agent(
                            agent_type=agent,
                            feature_mode=feature_mode,
                            reward_type=reward_type,
                            total_steps=total_steps,
                            train_end=train_end,
                            val_end=val_end
                        )
                        all_metrics = [train_metrics, eval_metrics, test_metrics]
                        self.__print_metrics_summary(all_metrics, f"{agent}-{feature_mode}-{reward_type}")
                    except Exception as e:
                        print(f"Failed to train {agent}-{feature_mode}-{reward_type}: {str(e)}")

        return all_metrics

    @staticmethod
    def __print_metrics_summary(metrics_list, filename):
        output_lines = []

        for idx, metrics in enumerate(metrics_list):
            if idx == 0:
                dataset_type = "TRAIN"
            elif idx == 1:
                dataset_type = "EVAL"
            else:
                dataset_type = "TEST"

            section = [
                f"=== {dataset_type} METRICS ===",
                "PERFORMANCE METRICS:",
                f"\tCumulative Return: {metrics['cumulative_return']:.2%}",
                f"\tAnnualized Return: {metrics['annualized_return']:.2%}",
                f"\tSharpe Ratio: {metrics['sharpe_ratio']:.2f}",
                f"\tMax Drawdown: {metrics['max_drawdown']:.2%}",
                f"\tVolatility: {metrics['volatility']:.2f}",
                f"\tTurnover: {metrics['turnover']:.2f}",
                "",
                "FINAL ALLOCATIONS:"
            ]

            # Add allocations
            for asset, alloc in metrics['final_allocations'].items():
                section.append(f"\t{asset}: {alloc:.1%}")

            # Add separator between sections
            section.extend(["", "=" * 60, ""])
            output_lines.extend(section)

        # Combine into single string
        output_str = "\n".join(output_lines)

        # Print to console
        print(output_str)

        # Save to experiment-specific file
        log_dir = Path(TEXT_PATH)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{filename}_metrics.txt"
        try:
            with open(log_file, "a") as f:
                f.write(output_str + "\n")
            print(f"Metrics saved to {log_file}")
        except Exception as e:
            print(f"Error saving metrics to file: {str(e)}")

