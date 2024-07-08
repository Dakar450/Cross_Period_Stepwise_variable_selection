from Simple_Rolling_Prediction import Simple_Rolling_Prediction
import pandas as pd
import numpy as np
import multiprocessing as mp
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from scipy import optimize as op
import warnings
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')

class Step_Forward_Strategy(Simple_Rolling_Prediction):
    def __init__(self):
        super().__init__()

    def imp_filter(self, num, model, data):
        imp = pd.Series(model.feature_importances_, index=data.columns)
        imp_sorted = imp.sort_values(ascending=False)
        fac_list = list(imp_sorted[0:num].index)
        return fac_list

    def length_chose(self, length, train_x0, train_y0, valid_len, valid_start):
        Times = self.time_label.copy()
        train_x, train_y = train_x0.loc[Times[valid_start-length:valid_start]], train_y0.loc[Times[valid_start-length:valid_start]]
        valid_x, valid_y = train_x0.loc[Times[valid_start:valid_start + valid_len]], train_y0.loc[Times[valid_start:valid_start + valid_len]]
        model = xgb.XGBRegressor(booster='gbtree', eta=0.3, eval_metric='rmse', max_depth=3, n_estimators=10)
        model.fit(train_x,train_y)
        y_pre = model.predict(valid_x)
        valid_set = self.y_concat(valid_y,y_pre,'valid_y','y_pred')
        cor_vali = valid_set.groupby('Time').corr(method = 'spearman').unstack()['valid_y']
        IC = cor_vali['y_pred'].mean()
        return [length, IC]

    def IC_forward(self, x_data, y_data, chosen, iternames, valid_start, train_len, valid_len):
        Times = self.time_label.copy()
        variables = []
        variables.extend(chosen)
        variables.append(iternames)
        x_sel = x_data[variables]
        train_x, train_y = x_sel.loc[Times[valid_start - train_len:valid_start]], y_data.loc[Times[valid_start - train_len:valid_start]]
        valid_x, valid_y = x_sel.loc[Times[valid_start:valid_start + valid_len]], y_data.loc[Times[valid_start:valid_start + valid_len]]
        model = xgb.XGBRegressor(booster='gbtree', eta=0.3, eval_metric='rmse', max_depth=3, n_estimators=10)
        model.fit(train_x, train_y)
        y_pred = model.predict(valid_x)
        valid_set = self.y_concat(valid_y, y_pred, 'valid_y', 'pred_y')
        cor_vali = valid_set.groupby('Time').corr(method='spearman').unstack()['valid_y']
        IC = cor_vali['pred_y'].mean()
        return [iternames, IC]

    def run_forward_strategy(self, start_date, train_len, vali_len, window_vali_len, window_list, pred_len, itertime, iterstep, num_imp, ratio, charge):
        Times = self.time_label.copy()
        vali_ICs, test_ICs = [], []
        asset_his, rate_his = [], []
        lengths = []
        asset = 1
        summary_forward = {}
        his_long, his_short = pd.Series([]), pd.Series([])
        for date in trange(start_date, len(Times)):
            IC_init = 0
            num = 0
            chosen = []
            train_x, train_y, test_x0, test_y = self.data_split(date, train_len, pred_len)
            model = xgb.XGBRegressor(booster='gbtree', eta=0.3, eval_metric='rmse', max_depth=3, n_estimators=10)
            model.fit(train_x, train_y)
            fac_list = self.imp_filter(num_imp, model, train_x)
            summary_forward[str(Times[date])] = []
            train_x = train_x[fac_list]
            for i in range(itertime):
                pool = mp.Pool(3)
                record = pool.starmap(self.IC_forward, [(train_x, train_y, chosen, itername, date - vali_len, train_len - vali_len, vali_len) for itername in fac_list])
                pool.close()
                record = pd.DataFrame(record, columns=["var_name", "mean_IC"])
                record = record.sort_values(by="mean_IC", ascending=False)
                choose = list(record["var_name"])[0:iterstep]
                for j in choose:
                    fac_list.remove(j)
                for s in range(iterstep):
                    num += 1
                    eachvar = self.IC_forward(train_x, train_y, chosen, choose[s], date - vali_len, train_len - vali_len, vali_len)
                    chosen.append(choose[s])
                    summary_forward[str(Times[date])].append([num, eachvar[0], eachvar[1]])
                IC_new = eachvar[1]
                IC_gain = IC_new - IC_init
                IC_init = IC_new
                if IC_gain < 0:
                    break
            vali_ICs.append(IC_init)
            train_x = train_x[chosen]
            test_x = test_x0[chosen]
            pool = mp.Pool(3)
            record1 = pool.starmap(self.length_chose,[(length, train_x, train_y, window_vali_len, date - 1) for length in window_list])
            pool.close()
            record1 = pd.DataFrame(record1, columns=["length", "mean_IC"])
            record1 = record1.sort_values(by="mean_IC", ascending=False)
            chose_length = list(record1['length'])[0]
            # chose_length = 20
            train_x, train_y = train_x.loc[Times[date - chose_length:date]], train_y.loc[
                Times[date - chose_length:date]]
            model = xgb.XGBRegressor(booster='gbtree', eta=0.3, eval_metric='rmse', max_depth=3, n_estimators=10)
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            test_set = self.y_concat(test_y, y_pred, 'test_y', 'y_pred')
            cor_vali = test_set.groupby('Time').corr(method='spearman').unstack()['test_y']
            test_IC = cor_vali['y_pred'].mean()
            his_long, his_short, rate, asset = self.long_short(test_set, asset, ratio, charge, his_long, his_short)
            lengths.append(chose_length)
            test_ICs.append(test_IC)
            asset_his.append(asset)
            rate_his.append(rate)
            print(Times[date], chose_length, IC_init, test_IC, rate, asset)

        print("mean rank IC: ", np.mean(test_ICs))
        print("ICIR: ", np.mean(test_ICs) / np.std(test_ICs) * np.sqrt(len(test_ICs)))
        print(np.mean(rate_his) / np.std(rate_his) * np.sqrt(52))
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
        print("end of the step-forward factor selection strategy")
        return rate_his
