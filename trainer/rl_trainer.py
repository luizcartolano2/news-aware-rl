from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.portfolio_env import PortfolioEnv
from dataset.rl_dataset import RLDataset


class RLTrainer:
    def __init__(
        self,
        price_file: str,
        macro_file: str,
        sentiment_file: str,
        reward_columns: list,
        reward_type: str = "log_return",
        feature_mode: str = "all",
        model_type: str = "PPO",
        normalize: bool = True,
        rolling_windows: list = [3, 7],
        window_size: int = 3,
        reward_horizon: int = 1,
        train_end: str = "2017-01-01",
        val_end: str = "2019-01-01",
        verbose: int = 1,
    ):
        self.model_type = model_type
        self.verbose = verbose
        self.train_end = train_end
        self.val_end = val_end

        # Store arguments
        self.dataset_args = dict(
            price_file=price_file,
            macro_file=macro_file,
            sentiment_file=sentiment_file,
            normalize=normalize,
            feature_mode=feature_mode,
            reward_columns=reward_columns,
            window_size=window_size,
            rolling_windows=rolling_windows,
            reward_horizon=reward_horizon,
            reward_type=reward_type,
        )

        # Load dataset and split
        self.dataset = RLDataset(**self.dataset_args)
        self.x_train, _, _, self.y_train, _, _ = self.dataset.split(self.train_end, self.val_end)

        # Build environment
        self.env = DummyVecEnv([self._make_env])  # wrapped for SB3 compatibility

    def _make_env(self):
        x_train, _, _, y_train, _, _ = self.dataset.split(
            train_end=self.train_end,
            val_end=self.val_end,
            output="np"
        )
        return Monitor(PortfolioEnv(x=x_train, y=y_train))

    def get_model(self):
        # Pick model class
        models = {
            "PPO": PPO,
            "A2C": A2C,
            "DDPG": DDPG,
        }
        model_class = models.get(self.model_type.upper())
        if model_class is None:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model_class("MlpPolicy", self.env, verbose=self.verbose)

    def train(self, total_timesteps=10_000, eval_freq=2000):
        eval_env = self._make_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        self.model = self.get_model()
        self.model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    def evaluate(self, episodes: int = 1):
        env = self._make_env()
        for ep in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _states = self.model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            print(f"Episode {ep + 1}: reward = {total_reward}")

    def save(self, path="trained_model.zip"):
        self.model.save(path)
