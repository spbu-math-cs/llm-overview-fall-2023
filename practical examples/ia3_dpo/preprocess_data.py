from typing import Dict, List, Tuple

from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from constants import DATASET_NAME, SUBSTITUTION_MAP


def get_texts() -> Tuple[List[str], List[str]]:
    dataset = load_dataset(DATASET_NAME)

    ru_texts = dataset["ru_mnli"]["premise"]
    en_texts = dataset["ru_mnli"]["premise_original"]

    return ru_texts, en_texts


def replace_letters(data: List[str]) -> List[str]:
    preprocessed_data = []

    for str_ in tqdm(data):
        preprocessed_str = ""
        for char in str_:
            preprocessed_str += SUBSTITUTION_MAP.get(char, char)

        preprocessed_data.append(preprocessed_str)

    return preprocessed_data


class PaddingDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.LongTensor]:
        padded_batch = {}
        
        for key in features[0].keys():
            to_pad = [torch.LongTensor(item[key]) for item in features]
            
            if key.endswith("_input_ids"):
                padding_value = self.pad_token_id
            else:
                padding_value = 0

            padded_batch[key] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

        return padded_batch
