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
    visual_output = None

    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()

        self.df = self._change_prices(df)
        self.reward_range = (0, M_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = sps.Box(
            low=numpy.array([0, 0]), high=numpy.array([3, 1]), dtype=numpy.float16)

        # The OHCL values for the last five prices are included in the prices
        self.observation_space = sps.Box(
            low=0, high=1, shape=(5, L_WINDOW_SIZE + 2), dtype=numpy.float16)

        self.balance = P_ACCOUNT_BALANCE
        self.n_worth = P_ACCOUNT_BALANCE  # Net Worth
        self.m_net_worth = P_ACCOUNT_BALANCE  # Maximum net Worth
        self.s_held = 0  # Share Held
        self.c_basis = 0  # Cost Basis
        self.t_shares_sold = 0  # Total shares sold
        self.t_sales_value = 0  # Total sales value
        self.c_transaction = 0  # Current Transaction
        self.trades = []
        self.profit = self.n_worth - P_ACCOUNT_BALANCE  # Current Profit

    def step(self, action):

        # Execute one time step(Transaction) within the environment
        self._perform_action(action)

        self.c_transaction += 1
        d_modifier = (self.c_transaction / M_STEPS)
        temp = self.balance * d_modifier
        reward = temp + self.c_transaction
        done = self.n_worth <= 0 or self.c_transaction >= len(self.df.loc[:, 'Open'].values)
        obs = self._next_inspection()

        return obs, reward, done, {}

    def reset(self):

        # Reset the environment to its original state.
        self.balance = P_ACCOUNT_BALANCE
        self.n_worth = P_ACCOUNT_BALANCE
        self.m_net_worth = P_ACCOUNT_BALANCE
        self.s_held = 0
        self.c_basis = 0
        self.t_shares_sold = 0
        self.t_sales_value = 0
        self.c_transaction = 0
        self.trades = []
        return self._next_inspection()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._write_output_to_file(kwargs.get('filename', 'output.txt'))

        elif mode == 'live':
            if self.visual_output is None:
                self.visual_output = StockTradingGraph(self.df, kwargs.get('title', None))

            if self.c_transaction > L_WINDOW_SIZE:
                self.visual_output.render(self.c_transaction, self.n_worth, self.trades, window_size=L_WINDOW_SIZE)

            self._write_output_to_terminal()

    def close(self):
        if self.visual_output != None:
            self.visual_output.close()
            self.visual_output = None

    def _change_prices(self, df):
        a_ratio = df['Adjusted_Close'] / df['Close']

        df['Open'] = df['Open'] * a_ratio
        df['High'] = df['High'] * a_ratio
        df['Low'] = df['Low'] * a_ratio
        df['Close'] = df['Close'] * a_ratio

        return df

    def _next_inspection(self):
        frame = numpy.zeros((5, L_WINDOW_SIZE + 1))

        # Scale the stock data points for the previous 5 days to between 0 and 1.
        numpy.put(frame, [0, 4], [
            self.df.loc[self.c_transaction: self.c_transaction + L_WINDOW_SIZE, 'Open'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction + L_WINDOW_SIZE, 'High'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction + L_WINDOW_SIZE, 'Low'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction + L_WINDOW_SIZE, 'Close'].values / M_SHARE_PRICE,
            self.df.loc[self.c_transaction: self.c_transaction + L_WINDOW_SIZE, 'Volume'].values / M_NUM_SHARES,
        ])

        # Add more information and scale each value from 0 to 1.
        obs = numpy.append(frame, [
            [self.balance / M_ACCOUNT_BALANCE],
            [self.m_net_worth / M_ACCOUNT_BALANCE],
            [self.s_held / M_NUM_SHARES],
            [self.c_basis / M_SHARE_PRICE],
            [self.t_sales_value / (M_NUM_SHARES * M_SHARE_PRICE)],
        ], axis=1)

        return obs

    def _perform_action(self, action):
        c_price = r.uniform(self.df.loc[self.c_transaction, "Open"], self.df.loc[self.c_transaction, "Close"])
        a_type = action[0]
        amount = action[1]

        if a_type < 1:
            # Purchase a percentage of the balance in stocks.
            t_possible = int(self.balance / c_price)
            s_bought = int(t_possible * amount)
            p_cost = self.c_basis * self.s_held  # Purchase Cost
            a_cost = s_bought * c_price  # Additional Cost

            self.balance -= a_cost
            self.c_basis = (p_cost + a_cost) / (self.s_held + s_bought)
            self.s_held += s_bought

            if s_bought > 0:
                self.trades.append({'step': self.c_transaction,
                                    'shares': s_bought, 'total': a_cost,
                                    'type': "buy"})

        elif a_type < 2:

            # Percentage of shares held for sale
            s_sold = int(self.s_held * amount)  # Shares Sold
            self.balance += s_sold * c_price
            self.s_held -= s_sold
            self.t_shares_sold += s_sold
            self.t_sales_value += s_sold * c_price

            if s_sold > 0:
                self.trades.append({'step': self.c_transaction,
                                    'shares': s_sold, 'total': s_sold * c_price,
                                    'type': "sell"})

        self.n_worth = self.balance + self.s_held * c_price

        if self.n_worth > self.m_net_worth:
            self.m_net_worth = self.n_worth

        if self.s_held == 0:
            self.c_basis = 0

    def _write_output_to_file(self, filename='output.txt'):

        file = open(filename, 'a+')

        file.write(f'Transaction: {self.c_transaction}\n')
        file.write(f'Remaining Balance: {self.balance}\n')
        file.write(f'Shares held: {self.s_held} (Total Shares sold: {self.t_shares_sold})\n')
        file.write(f'Average cost for held shares: {self.c_basis} (Total sales value: {self.t_sales_value})\n')
        file.write(f'Net worth: {self.n_worth} (Max net worth: {self.m_net_worth})\n')
        file.write(f'Total Profit: {self.profit}\n\n')

        file.close()

    def _write_output_to_terminal(self):

        print("---------------------------"f'Transaction: {self.c_transaction}'"------------------------------")
        print(f'Remaining Balance: {self.balance}\n')
        print(f'Shares held: {self.s_held} (Total Shares sold: {self.t_shares_sold})\n')
        print(f'Average cost for held shares: {self.c_basis} (Total sales value: {self.t_sales_value})\n')
        print(f'Net worth: {self.n_worth} (Max net worth: {self.m_net_worth})\n')
        print(f'Total Profit: {self.profit}\n\n')
        print("---------------------Transaction End-------------------------")
