# -*- coding: utf-8 -*-
import sys
import numpy
import pandas

sys.path.append("lib")
import strategy
import utils
import simulator
from loader import Loader, Index
from strategies.combination import CombinationStrategy
from dateutil.relativedelta import relativedelta

class Env:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.actions = (0, 1, 2)
        self.state_size = 25 # X日間の窓内のデータを扱う
        self.data = None
        self.reset()

    def reset(self):
        self.step = 0
        self.score = 0
        self.reward = 0
        self.terminal = False
        self.code = None
        self.position = simulator.Position()
        if self.data is None:
            self.create_data(self.start_date, self.end_date)
        self.columns = [
            "average_cross",
            "macd_cross",
            "rci_cross",
            "env12_cross",
            "env11_cross",
            "env09_cross",
            "env08_cross",
            "macd_trend",
            "macdhist_trend",
            "daily_average_trend",
            "weekly_average_trend",
            "volume_average_trend",
            "stages_trend",
            "stages_average_trend",
            "rci_trend",
            "rci_long_trend",
            "rising_safety_trend",
            "fall_safety_trend",
            "stages",
            "macd_stages",
            "macdhist_stages"
        ]
        self.state = numpy.zeros((len(self.columns), self.state_size))
        self.max_step = len(self.data)
        print("[Env] max_step: %s" % self.max_step)


    def load_stock(self, start_date, end_date):
        combination_setting = strategy.CombinationSetting()
        strategy_creator = CombinationStrategy(combination_setting)
        codes = []
        for date in utils.daterange(utils.to_datetime(start_date), utils.to_datetime(end_date)):
            codes = list(set(codes + strategy_creator.subject(utils.to_format(date))))
        data = None
        while data is None or len(data) <= self.state_size:
            self.code = numpy.random.choice(codes)
            data = Loader.load(self.code, utils.to_format(utils.to_datetime(start_date) - relativedelta(months=12)), end_date)
            data = utils.add_stats(data)
            data = Loader.filter(data, start_date, end_date)
        print("code: [%s]" % self.code)
        return data

    def load_index(self, start_date, end_date):
        index = Index()
        data = Loader.load_index(index.nikkei ,start_date, end_date)
        data = utils.add_stats(data)
        return data

    def create_data(self, start_date, end_date):
        self.data = self.load_index(start_date, end_date)
        assert len(self.data) > self.state_size

    # アクションに応じて状態を変更する
    def update(self, action):
        self.step += 1
        self.reward = 0
        begin = self.step
        end = self.step + self.state_size

        current = self.data[begin:end]
        close = current["close"].iloc[-1]

        # 買い
        if action == 1:
            self.position.add_history(100, close)
        # 売り
        elif action == 2:
            hold = self.position.value()
            order = self.position.num()
            if order > 0:
                self.position.add_history(-order, close)
                gain = close - hold
                if gain > 0:
                    self.reward = int(order / 100)
                else:
                    self.reward = -int(order / 100) * 2
        elif action > 2:
            self.reward = 0

        self.state = self.screen(self.data[begin:end])
        self.terminal = False

        self.score += self.reward

        if self.max_step-1 <= end:
            self.terminal = True

    def execute_action(self, action):
        self.update(action)

    # 学習に使う状態データに変換する
    def screen(self, data):
        data = data.reset_index(drop=True)
        screen = numpy.zeros((len(self.columns), self.state_size))

        for i, column in enumerate(self.columns):
            screen[i] = data[column].as_matrix().tolist()

        return screen

    def observe(self):
        return self.state, self.reward, self.terminal

