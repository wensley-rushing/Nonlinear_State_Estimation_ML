# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:02:57 2022

@author: s163761
"""


#%% Imports
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.chdir("../../..")

#%% Define Model in Class 1
import torch
from torch import nn

class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()

        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        return self.sequential(x)


#Test that network works as intended ------------------------------------------
network = FullyConnectedModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
x = torch.rand(20, 5)
y = network(x)
print('Shape X: ', x.shape)
print('Shape Y: ', y.shape)

#%% Define Model in Class 2
from typing import Dict
from pytorch_forecasting.models import BaseModel


    
#%% Initialize model Data
# Below, we create such a dataset with 30 different observations - 10 for 3 time series.
import numpy as np
import pandas as pd

multi_target_test_data = pd.DataFrame(
    dict(
        target1=np.random.rand(30),
        target2=np.random.rand(30)*100,
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
    )
)
multi_target_test_data

#print(test_data)

#%% Initialize Data
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, TorchNormalizer

# create the dataset from the pandas dataframe
multi_target_dataset = TimeSeriesDataSet(
    multi_target_test_data,
    group_ids=["group"],
    target=["target1", "target2"],  # USING two targets
    time_idx="time_idx",
    min_encoder_length=5,
    max_encoder_length=5,
    min_prediction_length=2,
    max_prediction_length=2,
    time_varying_unknown_reals=["target1", "target2"],
    target_normalizer=MultiNormalizer(
        [EncoderNormalizer(), TorchNormalizer()]
    ),  # Use the NaNLabelEncoder to encode categorical target
)

print(multi_target_dataset.get_parameters())

#%% Convert data with DataLoader
# convert the dataset to a dataloader
dataloader = multi_target_dataset.to_dataloader(batch_size=4)

# and load the first batch
x, y = next(iter(dataloader))
print("x =", x)
print("\ny =", y)
print("\ny[0] =", y[0])  # target values are a list of targets

# print("\nsizes of x =")
# for key, value in x.items():
#     print(f"\t{key} = {value.size()}")


#%% Define new from_dataset
from typing import List, Union

from pytorch_forecasting.metrics import MAE, SMAPE, MultiLoss
from pytorch_forecasting.utils import to_list


class FullyConnectedMultiTargetModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        target_sizes: Union[int, List[int]] = [],
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size * len(to_list(self.hparams.target_sizes)),
            output_size=self.hparams.output_size * sum(to_list(self.hparams.target_sizes)),
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_cont"].size(0)
        network_input = x["encoder_cont"].view(batch_size, -1)
        prediction = self.network(network_input)
        # RESHAPE output to batch_size x n_decoder_timesteps x sum_of_target_sizes
        prediction = prediction.unsqueeze(-1).view(batch_size, self.hparams.output_size, sum(self.hparams.target_sizes))
        # RESHAPE into list of batch_size x n_decoder_timesteps x target_sizes[i] where i=1..len(target_sizes)
        stops = np.cumsum(self.hparams.target_sizes)
        starts = stops - self.hparams.target_sizes
        prediction = [prediction[..., start:stop] for start, stop in zip(starts, stops)]
        if isinstance(self.hparams.target_sizes, int):  # only one target
            prediction = prediction[0]

        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # We need to return a named tuple that at least contains the prediction.
        # The parameter can be directly forwarded from the input.
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        # By default only handle targets of size one here, categorical targets would be of larger size
        new_kwargs = {
            "target_sizes": [1] * len(to_list(dataset.target)),
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset.time_varying_known_categoricals) == 0
            and len(dataset.time_varying_known_reals) == 0
            and len(dataset.time_varying_unknown_categoricals) == 0
            and len(dataset.static_categoricals) == 0
            and len(dataset.static_reals) == 0
            and len(dataset.time_varying_unknown_reals)
            == len(dataset.target_names)  # Expect as as many unknown reals as targets
        ), "Only covariate should be in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)
    
#%% Model Prediction
model = FullyConnectedMultiTargetModel.from_dataset(
    multi_target_dataset,
    hidden_size=10,
    n_hidden_layers=2,
    loss=MultiLoss(metrics=[MAE(), SMAPE()], weights=[2.0, 1.0]),
)
#model.summarize("full")  # print model summary
print(model.hparams)

x, y = next(iter(dataloader))
yhat = model(x)

print(multi_target_dataset.x_to_index(x))

print('Y0: ', y[0])
print('YH: ',yhat[0])
