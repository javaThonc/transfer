#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.pytorch_gru import GRU
from qlib.contrib.data.handler import ALPHA360
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)
from qlib.utils import exists_qlib_data

# from qlib.model.learner import train_model
from qlib.utils import init_instance_by_config


if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
        from get_data import GetData

        GetData().qlib_data_cn(target_dir=provider_uri)

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    MARKET = "csi300"
    BENCHMARK = "SH000300"

    ###################################
    # train model
    ###################################
    DATA_HANDLER_CONFIG = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": MARKET,
        "infer_processors": [
            {"class": "Fillna",
             "kwargs": {
                 "fields_group": "feature",
             }},
        ],
        "learn_processors": [
            {"class": "DropnaLabel", }
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }


    task = {
        "model": {
            "class": "TabNet_Model",
            "module_path": "qlib.contrib.model.pytorch_tabnet",
            "kwargs": {
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                'handler': {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": DATA_HANDLER_CONFIG
                },
                'segments': {
                    'pretrain': ("2008-01-01", "2016-12-31"),
                    'pretrain_validation': ("2017-01-0", "2020-08-01"),
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                }
            }
        }
    }

    # model = train_model(task)
    model = init_instance_by_config(task['model'])
    dataset = init_instance_by_config(task['dataset'])
    
    #model.pretrain(dataset)
    model.fit(dataset, pretrain_file='./pretrain/best.model')

    pred_score = model.predict(dataset)

    
    # save pred_score to file
    pred_score_path = Path("~/tmp/qlib/pred_score.pkl").expanduser()
    pred_score_path.parent.mkdir(exist_ok=True, parents=True)
    pred_score.to_pickle(pred_score_path)

    ###################################
    # backtest
    ###################################
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
    }
    BACKTEST_CONFIG = {
        "verbose": False,
        "limit_threshold": 0.095,
        "account": 100000000,
        "benchmark": BENCHMARK,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }

    # use default strategy
    # custom Strategy, refer to: TODO: Strategy API url
    strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = normal_backtest(pred_score, strategy=strategy, **BACKTEST_CONFIG)

    ###################################
    # analyze
    # If need a more detailed analysis, refer to: examples/train_and_bakctest.ipynb
    ###################################
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"]
    )
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    print(analysis_df)


