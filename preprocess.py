"""
preprocess non-GP-interpolated data for use with learnable Fourier PE
"""

from functools import lru_cache
from functools import partial

from typing import Optional, Iterable

from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic, IterableSlice, PseudoShuffled
from gluonts.torch.util import IterableDataset

import torch
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, set_seed
from transformers.models.informer.modeling_informer import InformerMeanScaler, InformerStdScaler, InformerNOPScaler
from datasets import concatenate_datasets

import numpy as np
import pandas as pd
import pdb
from collections import Counter

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def normalize_all(example, field_name):
    # normalize to [0,1] overall (min-max normalization to get rid of negative values)
    values = example[field_name]
    example[field_name] = (values - np.min(values)) / (np.max(values) - np.min(values))
    return example

def normalize_by_channel(example, field_name):
    # normalize to [0,1] by channel (min-max normalization to get rid of negative values)
    for row in range(len(example[field_name])):
        min_value = np.min(example[field_name][row][example[field_name][row] != 0])
        row_values = example[field_name][row]
        example[field_name][row] = (row_values - min_value) / (np.max(row_values) - min_value)
    return example

def create_attention_mask(example):
    # create attention mask to ignore padding
    example["attention_mask"] = np.zeros_like(example["transposed_target"])
    example["attention_mask"][:, example['transposed_times_wv'][0] != 0] = 1 # mask if time value is 0 (padding)
    return example

def transform_start_field(batch, freq):
    # batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    #TODO: threw out start field, otherwise have to convert from mjd
    batch["start"] = [convert_to_pandas_period("2010-12-29", freq) for example in batch]
    return batch

def transform_raw_data_example(config, example):
    # was 300 x 2, need to be 2 x 300 (first dim is channel)
    example['transposed_target'] = np.array(example['target']).T
    example['transposed_times_wv'] = np.array(example['times_wv']).T
    # divide by max value to constrain to [0,1]
    example = create_attention_mask(example)
    example = normalize_by_channel(example, "transposed_times_wv")
    example = normalize_all(example, "transposed_target") # normalize flux, flux_err by max overall
    # masking comes after normalization bc mask value is 0
    # if config.mask:
    #     example = mask(example, config.mask_fraction if hasattr(config, "mask_fraction") else 0.5)
    return example

def transform_raw_data(dataset, config: PretrainedConfig):
    # normalize time, data; create attention mask
    dataset = dataset.map(partial(transform_raw_data_example, config))
    if not config.mask:
        dataset = dataset.add_column("start", [0] * len(dataset))
        dataset.set_transform(partial(transform_start_field, freq="1M"))
    # have to swap out these field names because can't change dataset field shapes in place
    dataset = dataset.remove_columns(["target", "times_wv"])
    dataset = dataset.rename_column("transposed_target", "target")
    dataset = dataset.rename_column("transposed_times_wv", "times_wv")

    # remove/rename fields
    name_mapping = {
                "times_wv": "time_features",
                FieldName.TARGET: "values",
                "attention_mask": "observed_mask",
            }

    # dataset['times_wv'] = np.asarray(dataset['times_wv'])
    # dataset['target'] = np.asarray(dataset['target'])
    # dataset['attention_mask'] = np.asarray(dataset['attention_mask'])
    dataset = dataset.rename_columns(name_mapping)
    dataset = dataset.with_format('np')

    return dataset

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    allow_padding: Optional[bool] = True,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if allow_padding else config.context_length,
            min_future=config.prediction_length,
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(
            min_past=0 if allow_padding else config.context_length,
            min_future=config.prediction_length
        ),
        "test": TestSplitSampler(),
    }[mode]

    print(f"instance splitter created with context length {config.context_length}, lags {config.lags_sequence}")

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader_raw(
    config: PretrainedConfig,
    dataset,
    time_features,
    seed,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: Optional[bool] = True,
    allow_padding: Optional[bool] = True,
    add_objid: Optional[bool] = False,
    **kwargs,
) -> Iterable:

    set_seed(seed)

    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")
        if "redshift" in dataset.column_names:
            dataset = dataset.rename_column("redshift", "static_real_features")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    if config.has_labels:
        TRAINING_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")
    elif config.mask:
        TRAINING_INPUT_NAMES.append("mask_label")

    transformed_data = transform_raw_data(dataset, config)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train", allow_padding) + SelectFields(
        TRAINING_INPUT_NAMES #+ ["objid"]
    )

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from all the possible transformed time series, 1 in our case)
    # randomly from within the target time series and return an iterator.
    training_instances = instance_splitter.apply(
        Cyclic(transformed_data)
        if shuffle_buffer_length is None
        else PseudoShuffled(
            Cyclic(transformed_data),
            shuffle_buffer_length=shuffle_buffer_length,
        )
    )

    # from the training instances iterator we now return a Dataloader which will
    # continue to sample random windows for as long as it is called
    # to return batch_size of the appropriate tensors ready for training!
    return IterableSlice(
        iter(
            DataLoader(
                IterableDataset(training_instances),
                batch_size=batch_size,
                **kwargs,
            )
        ),
        num_batches_per_epoch,
    )

def create_test_dataloader_raw(
    config: PretrainedConfig,
    dataset,
    time_features,
    seed,
    batch_size: int,
    allow_padding: Optional[bool] = True,
    add_objid: Optional[bool] = False,
    compute_loss: Optional[bool] = False,
    shuffle_buffer_length: Optional[int] = None,
    **kwargs,
):

    set_seed(seed)

    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")
        if "redshift" in dataset.column_names:
            dataset = dataset.rename_column("redshift", "static_real_features")

    if config.has_labels:
        PREDICTION_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    if compute_loss:
        PREDICTION_INPUT_NAMES += [
            "future_values",
            "future_observed_mask",
        ]

    transformed_data = transform_raw_data(dataset, config)
    # still passing in 'train' because otherwise it will take the last data point (no room for future values)
    instance_sampler = create_instance_splitter(config, "train", allow_padding) + SelectFields(
        PREDICTION_INPUT_NAMES
    )

    testing_instances = instance_sampler.apply(
        transformed_data
        if shuffle_buffer_length is None
        else PseudoShuffled(
            transformed_data,
            shuffle_buffer_length=shuffle_buffer_length,
        ),
        is_train=False
    )

    # This returns a Dataloader which will go over the dataset once.
    return DataLoader(
        IterableDataset(testing_instances), batch_size=batch_size, **kwargs
    )

