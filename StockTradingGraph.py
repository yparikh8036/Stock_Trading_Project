import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import style
from datetime import datetime

from mplfinance.original_flavor import candlestick_ohlc

style.use('dark_background')

V_CHART_HEIGHT = 0.50  # Volume chart height

U_COLOR = '#27A59A'
D_COLOR = '#EF534F'
U_TEXT_COLOR = '#73D3CC'
D_TEXT_COLOR = '#DC2C27'


class StockTradingGraph:

    def __init__(self, df, title=None):
        self.df = df
        df['dt'] = pandas.to_datetime(df['Date'])
        self.n_worth = numpy.zeros(len(df['Date']))  # Net Worth

        # Make a figure on the screen and give it a title.
        figure = plt.figure()
        figure.suptitle(title)

        # Make a top subplot for the axis of net worth.
        self.net_worth_axis = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)

        # Create a bottom subplot for the price/volume axis that is shared.
        self.price_axis = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_axis)

        # Create a new axis for volume that has the same x-axis as the price axis.
        self.volume_axis = self.price_axis.twinx()

        # To make the graph easier to read, add padding.
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Allow the graph to be displayed without interfering with the rest of the programme.
        plt.show(block=False)

    def render(self, c_step, n_worth, trades, w_size=40):
        self.n_worth[c_step] = n_worth

        w_start = max(c_step - w_size, 0)
        s_range = range(w_start, c_step + 1)

        # Format dates as timestamps, necessary for candlestick graph
        dates = numpy.array([convert_date_to_number(x) for x in self.df['Date'].values[s_range]])

        self._display_net_worth(c_step, n_worth, s_range, dates)
        self._display_price(c_step, n_worth, dates, s_range)
        self._display_volume(c_step, n_worth, dates, s_range)
        self._display_trades(c_step, trades, s_range)

        # To make the date ticks easier to read, format them.
        self.price_axis.set_xticklabels(self.df['Date'].values[s_range], rotation=45, horizontalalignment='right')

        # Remove duplicate net worth date labels from your spreadsheet.
        plt.setp(self.net_worth_axis.get_xticklabels(), visible=False)

        # Viewing frames before they are un-rendered is required.
        plt.pause(0.001)

    def close(self):
        plt.close()

    def _display_net_worth(self, c_step, n_worth, s_range, dates):

        # Clear the frame that was rendered in the previous step.
        self.net_worth_axis.clear()

        # Net worth of Plot
        self.net_worth_axis.plot_date(dates, self.n_worth[s_range], '-', label='Net Worth')

        # Show legend, which uses the plot label we created earlier.
        self.net_worth_axis.legend()
        leg = self.net_worth_axis.legend(loc=2, ncol=2, prop={'size': 8})
        leg.get_frame().set_alpha(0.4)

        l_date = convert_date_to_number(self.df['Date'].values[c_step])
        l_net_worth = self.n_worth[c_step]

        # On the net worth graph, annotate the current net worth.
        self.net_worth_axis.annotate('{0:.2f}'.format(n_worth), (l_date, l_net_worth),
                                     xytext=(l_date, l_net_worth),
                                     bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                     color="black",
                                     fontsize="small")

        # Include space above and below the minimum and maximum net worth.
        self.net_worth_axis.set_ylim(min(self.n_worth[numpy.nonzero(self.n_worth)]) / 1.25, max(self.n_worth) * 1.25)

    def _display_price(self, current_step, net_worth, dates, step_range):
        self.price_axis.clear()

        # Data for the OHCL candlestick graph should be formatted in this way.
        sticks = zip(dates,
                           self.df['Open'].values[step_range], self.df['Close'].values[step_range],
                           self.df['High'].values[step_range], self.df['Low'].values[step_range])

        # Plot the price using the mpl finance candlestick graph.
        candlestick_ohlc(self.price_axis, sticks, width=1, colorup=U_COLOR, colordown=D_COLOR)

        l_date = convert_date_to_number(self.df['Date'].values[current_step])
        l_close = self.df['Close'].values[current_step]
        l_high = self.df['High'].values[current_step]

        # Display the pricing axis, print the current price.
        self.price_axis.annotate('{0:.2f}'.format(l_close), (l_date, l_close),
                                 xytext=(l_date, l_high),
                                 bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                                 color="black",
                                 fontsize="small")

        # To make room for a volume chart, move the price axis up.
        ylim = self.price_axis.get_ylim()
        self.price_axis.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * V_CHART_HEIGHT, ylim[1])

    def _display_volume(self, current_step, net_worth, dates, step_range):
        self.volume_axis.clear()

        vol = numpy.array(self.df['Volume'].values[step_range])

        positive = self.df['Open'].values[step_range] - \
                         self.df['Close'].values[step_range] < 0
        negative = self.df['Open'].values[step_range] - \
                         self.df['Close'].values[step_range] > 0

        # The price direction on that date was used to colour the volume bars.
        self.volume_axis.bar(dates[positive], vol[positive], color=U_COLOR, alpha=0.4, width=1, align='center')
        self.volume_axis.bar(dates[negative], vol[negative], color=D_COLOR, alpha=0.4, width=1, align='center')

        # Ticks are hidden and the volume axis height is limited below the price chart.
        self.volume_axis.set_ylim(0, max(vol) / V_CHART_HEIGHT)
        self.volume_axis.yaxis.set_ticks([])

    def _display_trades(self, current_step, trades, step_range):
        for t in trades:
            if t['step'] in step_range:
                date = convert_date_to_number(self.df['Date'].values[t['step']])
                high = self.df['High'].values[t['step']]
                low = self.df['Low'].values[t['step']]

                if t['type'] == 'buy':
                    high_low = low
                    color = U_TEXT_COLOR
                else:
                    high_low = high
                    color = D_TEXT_COLOR

                total = '{0:.2f}'.format(t['total'])

                # Print the current price to the price axis
                self.price_axis.annotate(f'${total}', (date, high_low),
                                         xytext=(date, high_low),
                                         color=color,
                                         fontsize=8,
                                         arrowprops=(dict(color=color)))


def convert_date_to_number(date):
    number = dates.datestr2num(datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d'))
    return number
