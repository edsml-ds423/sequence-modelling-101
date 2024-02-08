""""""

import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from torchvision import models, transforms
import torchvision.transforms as transforms

from src.data_source import (get_subfolder_names_from_root,
                         get_wind_speed_data,
                         create_sequence_data,
                         )


class StormWindSpeedSequencedDataset(Dataset):
    def __init__(self, 
                 data_root_dir: str,
                 storms_to_exclude: list=[],
                 storm_name: str|None=None,
                 numeric_cols: list|None=None,
                 time_column: str="relative_time",
                 dt: int|None=1800,
                 input_sequence_length: int=4,
                 target_length: int=1,
                 data_split: str|None = "train",
                 data_splits: tuple[float, float, float] = (0.8, 0.2, 0.2),
                 transform: transforms.Compose | None = None,
                 random_seed: int = 42,
                 ):
        """Initialisation."""
        # inputs
        self.data_root_dir = data_root_dir
        self.storms_to_exclude = storms_to_exclude
        self.storm_name = storm_name
        # get list of storm name(s) to use.
        self.storm_names = self.get_storm_names()
        self.numeric_cols = numeric_cols
        self.time_column = time_column
        self.dt = dt
        self.data_split = data_split
        self.data_splits = data_splits
        self.transform = transform
        # sequence info
        self.input_sequence_length = input_sequence_length
        self.target_length = target_length
        # seed
        self.random_seed = random_seed
    
        self.df = get_wind_speed_data(data_dir=self.data_root_dir,
                                      storm_names=self.storm_names,
                                      numeric_cols=self.numeric_cols,
                                      time_column=self.time_column,
                                      dt=self.dt,
                                      )
        
        # create sequenced wind speed dataset
        self.df_sequence = create_sequence_data(df=self.df,
                                                storm_names=self.storm_names,
                                                input_length=self.input_sequence_length,
                                                target_length=self.target_length,
                                                )
        
        self.data = self.split_data()

    
    def get_storm_names(self):
        """"""
        if self.storm_name is not None:
            storm_names = [self.storm_name]
        else:
            storm_names = get_subfolder_names_from_root(root_dir=self.data_root_dir)
            if self.storms_to_exclude is not None:
                storm_names = [s for s in storm_names if s not in self.storms_to_exclude]
            
        return storm_names    

    def split_data(self):
        """"""
        if self.data_split is None:
            # ability to return all of the data if needed i.e inference.
            return self.df_sequence
        else:
            train_val, df_test = train_test_split(
            self.df_sequence,
            train_size=self.data_splits[0],
            random_state=self.random_seed,
            shuffle=False
            )

            df_train, df_val = train_test_split(
                train_val,
                test_size=self.data_splits[1],
                random_state=self.random_seed,
                shuffle=False,
            )
            if self.data_split == "train":
                return df_train
            elif self.data_split == "val":
                return df_val
            elif self.data_split == "test":
                return df_test
            else:
                raise ValueError("Need to input either 'train', 'val' or None.")

    def __len__(self):
        """"""
        return len(self.data)

    def __getitem__(self, idx):
        """"""
        item = self.data.iloc[idx].copy()

        # suppliamentary data
        storm = item[:1].values[0]
        ocean = item[1:2].values[0]

        # get speed and time features
        time_arr = np.array(item[2: (self.input_sequence_length+self.target_length)+2], dtype=np.int16).reshape(-1, 1) # time 
        speed_arr = np.array(item[(self.input_sequence_length+self.target_length)+2: ], dtype=np.float32).reshape(-1, 1) # wind speed

        # make into a torch tensor and separate out input and target (after transform)
        # _data = torch.from_numpy(np.array([time_arr, speed_arr]))
        _data = torch.from_numpy(np.concatenate([time_arr, speed_arr], axis=1).reshape(self.input_sequence_length+self.target_length, -1))
        if self.transform:
            _data = self.transform(_data)

        input, target =  _data[:-self.target_length, :], _data[-self.target_length:, :]
        # input dimensions (sequence_length, num_features/input_size) 

        return (storm, ocean), input, target