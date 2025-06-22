""" Reinforcement Learning Dataset Class
This class handles loading, merging, and normalizing datasets for reinforcement learning tasks.

It includes methods for splitting the dataset into training, validation, and test sets,
as well as retrieving the full dataset and its features.

Example usage:
```python
from dataset.rl_dataset import RLDataset
dataset = RLDataset(
    price_file="prices.csv",
    macro_file="macros.csv",
    sentiment_file="sentiment.csv",
    reward_column="ibov",
    window_size=3,
    rolling_windows=[3, 7],  # moving average over 3 and 7 days
)
train, val, test = dataset.split()
```
"""
from typing import Literal, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from constants import DATA_PATH

# Define paths for datasets
DATA_PATH = f"{DATA_PATH}/consolidated"
# Define feature modes for dataset processing
FeatureMode = Literal["prices", "prices+macros", "all"]


class RLDataset:
    """
    Reinforcement Learning Dataset Class
    This class loads and processes datasets for reinforcement learning tasks,
    including asset prices, macroeconomic indicators, and sentiment data.
    Attributes:
        price_path (str): Path to the asset prices CSV file.
        macro_path (str): Path to the macroeconomic indicators CSV file.
        sentiment_path (str): Path to the daily sentiment CSV file.
        normalize (bool): Whether to normalize the features.
        feature_mode (FeatureMode): Mode for selecting features to include in the dataset.
        df (pd.DataFrame): Merged DataFrame containing all datasets.
        features (list): List of feature column names.
        scaler (StandardScaler): Scaler for normalizing features if `normalize` is True.
        reward_columns (Optional[str]): Column name for the reward signal, e.g., 'ibov' or 'brl_usd'.
        reward_horizon (int): Number of steps ahead to compute the reward.
        reward_type (Literal["log_return", "sharpe", "sortino", "volatility_penalty"]):
        window_size (int): Size of the time window for lagged features (1 = no lags, 3 = t-2, t-1, t).
        rolling_windows (Optional[List[int]]): List of rolling window sizes for additional features.
    """
    def __init__(
            self,
            price_file,
            macro_file,
            sentiment_file,
            normalize=True,
            feature_mode: FeatureMode = "all",
            reward_columns: Optional[List[str]] = None,
            reward_horizon: int = 1,
            reward_type: Literal["log_return", "sharpe", "sortino", "volatility_penalty"] = "log_return",
            window_size: int = 1,  # 1 = no lags, 3 = t-2, t-1, t
            rolling_windows: Optional[List[int]] = None,
    ):
        # Initialize paths and parameters
        self.price_path = f"{DATA_PATH}/{price_file}"
        self.macro_path = f"{DATA_PATH}/{macro_file}"
        self.sentiment_path = f"{DATA_PATH}/{sentiment_file}"
        self.normalize = normalize
        self.feature_mode = feature_mode
        self.window_size = window_size
        self.reward_columns = reward_columns or []  # e.g., 'ibov' or 'brl_usd'
        self.reward_horizon = reward_horizon
        self.reward_type = reward_type
        self.rolling_windows = rolling_windows

        # Load and merge datasets
        self.df = self.__load_and_merge()
        self.scaler = StandardScaler() if normalize else None
        self.features = [col for col in self.df.columns if col != "date"]

    @staticmethod
    def __add_lags(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Adds lagged features to the DataFrame based on the specified window size.
        This method creates lagged versions of the features by shifting them
        by the specified number of time steps (window size).
        :param df: a DataFrame containing date and feature columns.
        :param window: an integer representing the number of time steps to lag.
        :return: a DataFrame with added lagged features.
        """
        if window <= 1:
            return df
        lagged = [df[["date"]].copy()]

        for i in range(window):
            shifted = df.drop(columns=["date"]).shift(i)
            shifted.columns = [f"{col}_t-{i}" for col in shifted.columns]
            lagged.append(shifted)
        combined = pd.concat(lagged, axis=1).dropna().reset_index(drop=True)
        return combined

    @staticmethod
    def __add_reward(
            df: pd.DataFrame,
            assets: List[str],
            horizon: int = 1,
            reward_type: Literal["log_return", "sharpe", "sortino", "volatility_penalty"] = "log_return",
            rolling_window: int = 10,
            alpha: float = 0.1,
            minimum_std: float = 1e-6,
    ) -> pd.DataFrame:
        """
        Adds reward columns to the DataFrame based on specified reward formulation.

        :param df: DataFrame with asset prices.
        :param assets: List of asset column names to compute returns for.
        :param horizon: Steps ahead for future return.
        :param reward_type: Type of reward to compute.
        :param rolling_window: Rolling window for std in risk-adjusted returns.
        :param alpha: Penalty weight (for volatility_penalty).
        :param minimum_std: Minimum std to avoid division by zero.
        :return: DataFrame with added reward columns.
        """
        df = df.copy()

        for asset in assets:
            # Compute log return
            log_ret = np.log(df[asset]) - np.log(df[asset].shift(horizon))

            if reward_type == "log_return":
                reward = log_ret

            elif reward_type == "sharpe":
                rolling_std = log_ret.rolling(rolling_window).std().clip(lower=minimum_std)
                reward = log_ret / rolling_std

            elif reward_type == "sortino":
                downside = log_ret.copy()
                downside[downside > 0] = 0
                rolling_std_neg = downside.rolling(rolling_window).std().clip(lower=minimum_std)
                reward = log_ret / rolling_std_neg

            elif reward_type == "volatility_penalty":
                rolling_std = log_ret.rolling(rolling_window).std().fillna(0)
                reward = log_ret - alpha * rolling_std

            else:
                raise ValueError(f"Unsupported reward_type: {reward_type}")

            df[f"{asset}_reward_t+{horizon}"] = reward

        return df.dropna()

    def __add_rolling_statistics(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Adds rolling mean and standard deviation features for the specified windows.
        :param df: Input DataFrame with date and feature columns.
        :param windows: List of window sizes to compute rolling statistics on.
        :return: DataFrame with added rolling features.
        """
        df = df.copy()
        x_cols = [col for col in df.columns if
                  col not in ["date"] + [f"{r}_return_t+{self.reward_horizon}" for r in self.reward_columns]]

        for window in windows:
            for col in x_cols:
                df[f"{col}_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_std_{window}"] = df[col].rolling(window).std()
        return df

    def __load_and_merge(self) -> pd.DataFrame:
        """
        Load and merge datasets from specified paths.
        The datasets include asset prices, macroeconomic indicators, and sentiment data.
        The merging is done on the "date" column, and the resulting DataFrame is sorted by date.
        The method handles different feature modes to include or exclude certain datasets.
        :return: Merged DataFrame containing all datasets with date and feature columns.
        """
        # Load prices dataset
        prices = pd.read_csv(self.price_path, parse_dates=["date"])

        # Prepare a list of DataFrames to merge based on feature mode
        dfs = [prices]
        if self.feature_mode in ["prices+macros", "all"]:
            macro = pd.read_csv(self.macro_path, parse_dates=["date"])
            dfs.append(macro)
        if self.feature_mode == "all":
            sentiment = pd.read_csv(self.sentiment_path, parse_dates=["date"])
            dfs.append(sentiment)

        # Merge all DataFrames on "date"
        df = dfs[0]
        for other in dfs[1:]:
            df = pd.merge(df, other, on="date", how="inner")

        # Sort by date and drop rows with NaN values
        df = df.sort_values("date").dropna().reset_index(drop=True)

        # Add rolling statistics if specified
        if self.rolling_windows:
            df = self.__add_rolling_statistics(df, self.rolling_windows)

        # Add lagged features and reward column if specified
        df = self.__add_lags(df, self.window_size)

        if self.reward_columns:
            df = self.__add_reward(df, self.reward_columns, self.reward_horizon)

        return df

    def split(self, train_end: str, val_end: str, output: Literal["df", "np"] = "np") -> Tuple:
        """
        Split the dataset into training, validation, and test sets based on date ranges.
        The method normalizes the features if `normalize` is set to True.
        :param train_end: a string representing the end date for the training set (inclusive).
        :param val_end: a string representing the end date for the validation set (exclusive).
        :param output: a string indicating the output format, either "df" for DataFrames or "np" for NumPy arrays.
        :return: a tuple containing the training, validation, and test sets.
        """
        df = self.df.copy()

        train = df[df["date"] < train_end]
        val = df[(df["date"] >= train_end) & (df["date"] < val_end)]
        test = df[df["date"] >= val_end]

        x_cols = [col for col in train.columns if
                  col not in ["date"] + [f"{r}_reward_t+{self.reward_horizon}" for r in self.reward_columns]]
        y_cols = [f"{r}_reward_t+{self.reward_horizon}" for r in self.reward_columns]

        x_train, x_val, x_test = train[x_cols], val[x_cols], test[x_cols]
        y_train, y_val, y_test = train[y_cols], val[y_cols], test[y_cols]

        if self.normalize:
            self.scaler.fit(x_train[x_cols])
            x_train = self.scaler.transform(x_train[x_cols])
            x_val = self.scaler.transform(x_val[x_cols])
            x_test = self.scaler.transform(x_test[x_cols])

        if output == "df":
            return train[x_cols], val[x_cols], test[x_cols], y_train, y_val, y_test
        return x_train, x_val, x_test, y_train.values, y_val.values, y_test.values

    def get_full_data(self, normalized: bool = True) -> pd.DataFrame:
        """
        Retrieve the full dataset with optional normalization.
        If `normalized` is True, the features are scaled using the StandardScaler.
        :param normalized: a boolean indicating whether to return normalized features.
        :return: a DataFrame containing the full dataset with features.
        """
        # Check if normalization is required
        if not normalized or not self.normalize:
            return self.df.copy()

        # Normalize the features using StandardScaler
        scaled = self.scaler.fit_transform(self.df[self.features])
        df_scaled = pd.DataFrame(scaled, columns=self.features)
        df_scaled["date"] = self.df["date"].values
        return df_scaled

    def get_columns(self) -> list:
        """
        Retrieve the list of feature columns in the dataset.
        :return: a list of feature column names.
        """
        return self.features

    def get_scaler(self):
        """
        Retrieve the StandardScaler used for normalizing features.
        :return: a StandardScaler instance if normalization is enabled, otherwise None.
        """
        return self.scaler
