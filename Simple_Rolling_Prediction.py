import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')


class Simple_Rolling_Prediction:
    def __init__(self):
        self.data = None
        self.time_label = None

    def create(self, data):
        data['Time'] = pd.to_datetime(data['close_time'], format='%Y-%m-%d')
        data.set_index('Time', inplace=True)
        data = data.set_index(['jj_code'], append=True)
        data.drop(columns="close_time", inplace=True)
        data.sort_index(level=0, ascending=True, inplace=True)
        data['y_rank'] = data.groupby('Time')['y'].rank()
        self.data = data
        self.time_label = self.data.index.get_level_values(0).unique()
        return self

    # data split function
    def data_split(self, valid_start, train_len, valid_len):
        data, Times = self.data.copy(), self.time_label.copy()
        train_data = data.loc[Times[valid_start - train_len:valid_start]]
        valid_data = data.loc[Times[valid_start:valid_start + valid_len]]
        train_x = train_data[train_data.columns.drop(['y', 'y_rank'])]
        valid_x = valid_data[valid_data.columns.drop(['y', 'y_rank'])]
        train_y = train_data['y_rank']
        valid_y = valid_data['y']
        return train_x, train_y, valid_x, valid_y

    # data concat function in order to calculate IC conveniently
    def y_concat(self,y_true,y_pre,name0,name1):
        y_pre = pd.Series(y_pre, index=y_true.index)
        datacon = pd.concat([y_true, y_pre], axis=1)
        datacon.columns = [name0, name1]
        return datacon

    # calculate long short set based on the output factor
    def long_short(self, valid_data, asset, ratio, charge_ratio, his_long, his_short):
        valid_data.columns = ['valid_y', 'y_pred']
        valid_data.reset_index(level=0, inplace=True)
        valid_data.drop('Time', axis=1, inplace=True)
        valid_con = valid_data.sort_values(by='y_pred')
        nums = int(ratio * len(valid_con))
        short = valid_con.head(nums)
        long = valid_con.tail(nums)
        if his_long.empty or his_short.empty:
            print('start')
            his_long = pd.Series(0, index=long.index)
            his_short = pd.Series(0, index=long.index)
        long['weights'] = 0.5 / nums
        short['weights'] = 0.5 / nums
        index_long = long.index.union(his_long.index)
        index_short = short.index.union(his_short.index)
        long_change = long['weights'].reindex(index_long, fill_value=0) - his_long.reindex(index_long, fill_value=0)
        short_change = short['weights'].reindex(index_short, fill_value=0) - his_short.reindex(index_short, fill_value=0)
        weights_change = abs(long_change).sum() + abs(short_change).sum()
        rate = (long['valid_y'] * long['weights']).sum() - (
                    short['valid_y'] * short['weights']).sum() - charge_ratio * weights_change
        asset = asset * (1 + rate)
        return long['weights'], short['weights'], rate, asset

    # calculate rolling 1-year sharp ratio
    def Sharp_ratio1y(self, assets_return):
        Rp = assets_return.rolling(window=52).mean()
        sigma = assets_return.rolling(window=52).std() / np.sqrt(52)
        Sharp_ratio = Rp / sigma
        return Sharp_ratio.dropna()

    def run_strategy(self, start_date, train_len, pred_len, ratio, charge):
        data, Times = self.data.copy(), self.time_label.copy()
        train_ICs, vali_ICs = [], []
        asset = 1
        asset_his, rate_his = [], []
        his_long, his_short = pd.Series([]), pd.Series([])
        for date in trange(start_date,len(Times)):
            train_x,train_y,test_x,test_y = self.data_split(date, train_len, pred_len)
            model = xgb.XGBRegressor(booster='gbtree', eta=0.3, eval_metric='rmse', max_depth=3, n_estimators=10)
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            y_fit = model.predict(train_x)
            test_set = self.y_concat(test_y, y_pred, 'test_y', 'y_pred')
            cor_vali = test_set.groupby('Time').corr(method='spearman').unstack()['test_y']
            test_IC = cor_vali['y_pred'].mean()
            train_set = self.y_concat(train_y, y_fit, 'train_y', 'y_fit')
            train_cor = train_set.groupby('Time').corr(method='spearman').unstack()['train_y']
            train_IC = train_cor['y_fit'].mean()
            his_long, his_short, rate, asset = self.long_short(test_set, asset, ratio, charge, his_long, his_short)
            train_ICs.append(train_IC)
            vali_ICs.append(test_IC)
            asset_his.append(asset)
            rate_his.append(rate)
            print(Times[date], train_IC, test_IC, rate, asset)
        print("mean IC:", np.mean(vali_ICs), "IC_IR:", np.mean(vali_ICs) / np.std(vali_ICs) * np.sqrt(len(vali_ICs)))
        print("mean return", sum(rate_his) / len(rate_his))
        plt.plot(asset_his)
        plt.show()

        rate_his = pd.Series(rate_his, index=Times[start_date:])
        sharp_ratio1 = self.Sharp_ratio1y(rate_his)
        plt.plot(sharp_ratio1)
        plt.axhline(y=0, linestyle='--')
        plt.axhline(y=np.mean(sharp_ratio1))
        plt.show()
        print("mean rolling 1 year sharp ratio:", np.mean(sharp_ratio1))
        print("end of the simple rolling training and predicting strategy")
        return rate_his
