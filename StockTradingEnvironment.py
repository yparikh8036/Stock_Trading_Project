import random as r
import gym
from gym import spaces as sps
import numpy

from StockTradingGraph import StockTradingGraph

M_ACCOUNT_BALANCE = 1000000000  # Maximum account balance
M_NUM_SHARES = 1000000000  # Maximum number of Shares
M_SHARE_PRICE = 10000  # Maximum Share Price
M_OPEN_POSITIONS = 10  # Maximum open positions
M_STEPS = 50000  # Maximum Steps
P_ACCOUNT_BALANCE = 10000  # Primary account balance
L_WINDOW_SIZE = 40  # Lookback window size


def factor_pairs(val):
    return [(i, val / i) for i in range(1, int(val ** 0.5) + 1) if val % i == 0]


class StockTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()

        self.df = self._adjust_prices(df)
        self.reward_range = (0, M_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = sps.Box(
            low=numpy.array([0, 0]), high=numpy.array([3, 1]), dtype=numpy.float16)

        # The OHCL values for the last five prices are included in the prices
        self.observation_space = sps.Box(
            low=0, high=1, shape=(5, L_WINDOW_SIZE + 2), dtype=numpy.float16)

        self.balance = P_ACCOUNT_BALANCE
        self.net_worth = P_ACCOUNT_BALANCE
        self.max_net_worth = P_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.c_transaction = 0  # Current Transaction
        self.trades = []
        self.profit = self.net_worth - P_ACCOUNT_BALANCE

    def step(self, action):

        # Execute one time step(Transaction) within the environment
        self._take_action(action)

        self.c_transaction += 1

        delay_modifier = (self.c_transaction / M_STEPS)

        reward = self.balance * delay_modifier + self.c_transaction
        done = self.net_worth <= 0 or self.c_transaction >= len(
            self.df.loc[:, 'Open'].values)

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):

        # Reset the environment to its original state.
        self.balance = P_ACCOUNT_BALANCE
        self.net_worth = P_ACCOUNT_BALANCE
        self.max_net_worth = P_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.c_transaction = 0
        self.trades = []
        return self._next_observation()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._write_output_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(
                    self.df, kwargs.get('title', None))

            if self.c_transaction > L_WINDOW_SIZE:
                self.visualization.render(
                    self.c_transaction, self.net_worth, self.trades, window_size=L_WINDOW_SIZE)

            self._write_output_to_terminal()

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

    def _adjust_prices(self, df):
        adjust_ratio = df['Adjusted_Close'] / df['Close']

        df['Open'] = df['Open'] * adjust_ratio
        df['High'] = df['High'] * adjust_ratio
        df['Low'] = df['Low'] * adjust_ratio
        df['Close'] = df['Close'] * adjust_ratio

        return df

    def _next_observation(self):
        frame = numpy.zeros((5, L_WINDOW_SIZE + 1))

        # Get the stock data points for the last 5 days and scale to between 0-1
        numpy.put(frame, [0, 4], [
            self.df.loc[self.c_transaction: self.c_transaction +
                                            L_WINDOW_SIZE, 'Open'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction +
                                            L_WINDOW_SIZE, 'High'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction +
                                            L_WINDOW_SIZE, 'Low'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction +
                                            L_WINDOW_SIZE, 'Close'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction +
                                            L_WINDOW_SIZE, 'Volume'].values / M_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = numpy.append(frame, [
            [self.balance / M_ACCOUNT_BALANCE],
            [self.max_net_worth / M_ACCOUNT_BALANCE],
            [self.shares_held / M_NUM_SHARES],
            [self.cost_basis / M_SHARE_PRICE],
            [self.total_sales_value / (M_NUM_SHARES * M_SHARE_PRICE)],
        ], axis=1)

        return obs

    def _take_action(self, action):
        current_price = r.uniform(
            self.df.loc[self.c_transaction, "Open"], self.df.loc[self.c_transaction, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                                      prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought > 0:
                self.trades.append({'step': self.c_transaction,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "buy"})

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'step': self.c_transaction,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def _write_output_to_file(self, filename='render.txt'):

        file = open(filename, 'a+')

        file.write(f'Transaction: {self.c_transaction}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {self.profit}\n\n')

        file.close()

    def _write_output_to_terminal(self):

        print("---------------------------"f'Transaction: {self.c_transaction}'"------------------------------")
        print(f'Balance: {self.balance}\n')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        print(f'Profit: {self.profit}\n\n')
        print("---------------------Transaction End-------------------------")
