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
        "fit_start_time":"2008-01-01",
        "fit_end_time":"2014-12-31",
        "instruments": MARKET,
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
                    'pretrain': ("2008-01-01", "2020-08-01")
                }
            }
        }
    }

    # model = train_model(task)
    model = init_instance_by_config(task['model'])
    dataset = init_instance_by_config(task['dataset'])
    model.pretrain(dataset)
