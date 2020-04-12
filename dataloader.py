# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser
from torchnlp.datasets.dataset import Dataset


def collate_lists(text: list) -> dict:
    """ Converts each line into a dictionary. """
    collated_dataset = []
    for i in range(len(text)):
        collated_dataset.append({"text": str(text[i])})
    return collated_dataset


def text_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def load_dataset(path):
        df = pd.read_csv(path)
        text = list(df.text)
        return Dataset(collate_lists(text))

    func_out = []
    if train:
        func_out.append(load_dataset(hparams.train_csv))
    if val:
        func_out.append(load_dataset(hparams.dev_csv))
    if test:
        func_out.append(load_dataset(hparams.test_csv))

    return tuple(func_out)
