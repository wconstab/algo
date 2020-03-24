import pandas as pd
import numpy as np

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.application.handlers.handler import Handler
from bokeh.application.application import Application
from bokeh.server.server import Server
from bokeh.palettes import Viridis
from threading import Lock, Thread


class BokehPlotter(object):
    def __init__(self, handlers, port=5006):
        self.app = Application()
        for handler in handlers:
            self.app.add(handler)
        self.server = Server({'/': self.app}, port=port)
        self.thread = Thread(target=self.server.io_loop.start)

    def start(self):
        self.server.start()
        self.server.io_loop.add_callback(self.server.show, '/')
        self.thread.start()

    def stop(self):
        self.server.io_loop.stop()
        self.server.stop()
        self.thread.join()

class CostPlotHandler(Handler):
    def __init__(self, costs=['train', 'val'], palette=Viridis, n_show=1000):
        super(CostPlotHandler, self).__init__()
        self.period_ms = 500
        self.update = False
        self.costs = costs
        self.columns = ['iter'] + costs
        self.colors = palette[max(3, len(costs))][:len(costs)]
        self.df = pd.DataFrame(columns=self.columns)
        self.lock = Lock()
        self.n_show = 1000

    def modify_document(self, doc):
        self.source = ColumnDataSource(self.df)
        p = figure(plot_height=250)
        p.xaxis.axis_label = 'Iteration'
        p.yaxis.axis_label = 'Cost'
        p.y_range.start = 0
        p.y_range.end = 3

        for cost, color in zip(self.costs, self.colors):
            p.line('iter', cost, color=color, source=self.source, alpha=0.7)
            p.circle('iter', cost, color=color, source=self.source, alpha=0.7, legend_label=cost)
        p.legend.location = "top_right"
        p.legend.click_policy="hide"
        def callback():
            with self.lock:
                if self.update:
                    self.source.stream(self.df, self.n_show)
                    self.update = False

        doc.add_root(p)
        doc.add_periodic_callback(callback, self.period_ms)

    def add_data(self, update_dict):
        with self.lock:
            df = pd.DataFrame(update_dict, columns=self.columns)
            self.df = self.df.append(df)
            self.update = True
    


class PeriodicDataframeHandler(Handler):
    def __init__(self):
        super(PeriodicDataframeHandler, self).__init__()
        self.period_ms = 500
        self.update = False
        self.df = pd.DataFrame(columns=['x', 'y'])
        self.reset = False
        self.lock = Lock()
        self.n_show = 100
        
    def modify_document(self, doc):
        self.source = ColumnDataSource(self.df)
        p = figure()
        p.scatter('x', 'y', source=self.source)
        def callback():
            with self.lock:
                if self.reset:
                    self.reset = False
                    self.source.data = ColumnDataSource.from_df(self.df)
                if self.update:
                    self.source.stream(self.df, self.n_show)
                    self.update = False

        doc.add_root(p)
        doc.add_periodic_callback(callback, self.period_ms)

    def add_data(self, new_df_or_update):
        with self.lock:
            df = pd.DataFrame(new_df_or_update, columns=['x', 'y'])
            self.df = self.df.append(df)
            self.update = True
        
    def reset_data(self):
        with self.lock:
            self.df = pd.DataFrame(columns=['x', 'y'])
            self.reset = True