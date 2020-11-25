# backtest by pred_score.pkl
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import REG_CN
#from qlib.contrib.model.pytorch_gru import GRU
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
    pred_score = pd.read_pickle('./pred_freq25out32.pkl')

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
