import pandas as pd
import numpy as np

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.application.handlers.handler import Handler
from bokeh.application.application import Application
from bokeh.server.server import Server
from threading import Lock, Thread


class BokehPlotter(object):
    def __init__(self, port=5006):
        self.handler = PeriodicDataframeHandler()
        # self.server = Server(Application(self.handler), io_loop=IOLoop(), port=5006)
        self.server = Server({'/': Application(self.handler)}, port=port)
        self.thread = Thread(target=self.server.io_loop.start)

    def start(self):
        self.server.start()
        self.server.io_loop.add_callback(self.server.show, '/')
        self.thread.start()

    def stop(self):
        self.server.io_loop.stop()
        self.server.stop()
        self.thread.join()
    
    def add_data(self, data):
        self.handler.add_data(data)


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