from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

from StockTradingEnvironment import StockTradingEnvironment

import pandas

data = pandas.read_csv('./data/NVDA.csv')
data = data.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnvironment(data)])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50)

obs = env.reset()

for i in range(len(data['Date'])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='live')
