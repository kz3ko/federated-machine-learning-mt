from os import makedirs
from random import shuffle
from typing import Union
from json import loads


def get_data_from_json(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return loads(f.read())


def get_shuffled_data(data: Union[list, set]) -> Union[list, set]:
    shuffle(data)
    return data


def create_directory(path: str):
    makedirs(path, exist_ok=True)
