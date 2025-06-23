from rl.portfolio_trainer import PortfolioTrainer

if __name__ == "__main__":
    trainer = PortfolioTrainer(
        price_file="asset_prices.csv",
        macro_file="macro_indicators.csv",
        sentiment_file="daily_sentiments.csv",
        reward_columns=["ibov_t-0", "brl_usd_t-0"],
    )

    results = trainer.run_experiments(
        agent_types=["ppo", "a2c"],
        feature_modes=["prices", "prices+macros", "all"],
        reward_types=["log_return", "sharpe", "sortino", "volatility_penalty"],
        train_end="2018-01-01",
        val_end="2019-01-01",
        total_steps=10_000,
    )


