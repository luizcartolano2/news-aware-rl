from trainer.rl_trainer import RLTrainer

trainer = RLTrainer(
    price_file="asset_prices.csv",
    macro_file="macro_indicators.csv",
    sentiment_file="daily_sentiments.csv",
    reward_columns=["ibov_t-0", "brl_usd_t-0"],
    reward_type="volatility_penalty",
    model_type="PPO",
)

trainer.train(total_timesteps=10000)
trainer.evaluate()
trainer.save()
