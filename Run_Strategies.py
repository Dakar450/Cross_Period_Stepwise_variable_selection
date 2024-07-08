import pandas as pd
import seaborn as sns
from Simple_Rolling_Prediction import Simple_Rolling_Prediction
from Step_Forward_Strategy import Step_Forward_Strategy
import warnings
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')

if __name__ == "__main__":
    data = pd.read_csv('../Factor_info/factors_non_BTC.csv')
    params = {
        "data": data
    }
    run_params = {
        "start_date": 75,
        "train_len": 20,
        "pred_len": 1,
        "ratio": 0.2,
        "charge": 0.0005
    }
    Simple_Strategy = Simple_Rolling_Prediction().create(**params)
    simple_strategy_return = Simple_Strategy.run_strategy(**run_params)
    simple_strategy_return.to_csv('simple_rolling_prediction.csv',header=True)
    '''
    Complex_Strategy = Step_Forward_Strategy().create(**params)
    run_complex_params = {
        "start_date": 75,
        "train_len": 20,
        "vali_len":4,
        "window_vali_len":2,
        "window_list":[20,16,12,8,4],
        "pred_len": 1,
        "itertime":5,
        "iterstep":3,
        "num_imp":60,
        "ratio": 0.2,
        "charge": 0.0005,

    }
    Complex_Strategy_return = Complex_Strategy.run_forward_strategy(**run_complex_params)
    '''