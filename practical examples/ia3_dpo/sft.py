import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.model_selection import train_test_split

from constants import (
    RANDOM_STATE,
    SFT_TRAIN_SIZE_RATIO,
    TOKENIZED_TEXT_MAX_LEN,
    MODEL_NAME,
    SFT_LEARNING_RATE,
    EPOCH_NUM,
    SAVE_STEPS,
    INSTRUCTION_PROMPT,
    BATCH_SIZE,
)
from ia3 import enable_adapters, wrap_model
from preprocess_data import PaddingDataCollator, get_texts, replace_letters
from utils import train_loop


def get_sft_data(tokenizer: LlamaTokenizer) -> List[str]:
    ru_texts, en_texts = get_texts()
    ru_texts_sft, _, en_texts_stf, _ = train_test_split(
        ru_texts,
        en_texts,
        train_size=SFT_TRAIN_SIZE_RATIO,
        random_state=RANDOM_STATE,
    )

    texts_sft = list(
        filter(
            lambda text: len(tokenizer.encode(text)) < TOKENIZED_TEXT_MAX_LEN,
            ru_texts_sft + en_texts_stf,
        )
    )

    return texts_sft


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[str], prompt: str, tokenizer: LlamaTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt = prompt

        # hardcode only for Llama2, see: 
        self.max_length = 4096
        self.max_prompt_length = self.max_length // 2

        self.data = data
        self.preprocessed_data = replace_letters(data)

    def __len__(self) -> int:
        return len(self.data)

    def build_tokenized_answer(self, prompt: str, answer: str) -> Dict[str, List[int]]:
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
    
        full_input_ids = np.array(full_tokenized["input_ids"])
    
        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length")
        
        response_token_ids_start_idx = len(prompt_input_ids)
    
        # If tokenized prompt is different than both prompt + answer, then it means the last token has changed due to merging
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1
    
        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]
    
        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]
    
        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "input_ids": answer_input_ids,
            "attention_mask": answer_attention_mask,
        }
    
    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        output = {}

        prompt = self.prompt.format(self.data[index])
        item = self.preprocessed_data[index]

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        item_tokens = self.build_tokenized_answer(prompt, item)
    
        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        item_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + item_tokens["prompt_input_ids"]
    
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        item_tokens["prompt_attention_mask"] = [1] + item_tokens["prompt_attention_mask"]
    
        # add EOS token to end of answer
        item_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        item_tokens["attention_mask"].append(1)
    
        response_length = len(item_tokens["input_ids"])
    
        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [item_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + response_length > self.max_length:
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][:self.max_prompt_length]
    
        # if that's still too long, truncate the response
        if len(item_tokens["prompt_input_ids"]) + response_length > self.max_length:
            for k in ["input_ids", "attention_mask"]:
                item_tokens[k] = item_tokens[k][:self.max_length - self.max_prompt_length]

        item_sequence_tokens = {k: item_tokens[f"prompt_{k}"] + item_tokens[k] for k in ["input_ids", "attention_mask"]}

        for k, toks in {"": item_sequence_tokens}.items():
            for type_key, tokens in toks.items():
                output[f"{k}{type_key}"] = tokens

        return output


def sft_loss(
    model: LlamaForCausalLM,
    batch: Dict[str, torch.LongTensor],
):
    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
    ).logits[:, :-1, :]

    labels = batch["input_ids"][:, 1:]
    
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def sft_train(
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = SFT_LEARNING_RATE,
    epoch_num: int = EPOCH_NUM,
    save_steps: int = SAVE_STEPS,
):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(model_name, device_map=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # make dataloader
    sft_data = get_sft_data(tokenizer)
    sft_train_dataset = SFTDataset(
        sft_data,
        INSTRUCTION_PROMPT,
        tokenizer,
    )
    sft_dataloader = torch.utils.data.DataLoader(
        sft_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PaddingDataCollator(tokenizer.pad_token_id),
    )

    # load model
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        low_cpu_mem_usage=True, 
        offload_state_dict=True,
        torch_dtype=torch.float32,
    )

    for param in model.parameters():
        param.requires_grad = False

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # wrap model with IA^3 adapters
    wrap_model(model)

    model.train()
    enable_adapters(model)

    sft_optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], 
        lr=learning_rate,
    )

    train_loop(
        model, 
        sft_optimizer, 
        sft_dataloader, 
        sft_loss,
        "sft",
        epoch_num=epoch_num,
        save_steps=save_steps,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=SFT_LEARNING_RATE,
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=EPOCH_NUM,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=SAVE_STEPS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sft_train(
        args.model_name,
        args.batch_size,
        args.learning_rate,
        args.epoch_num,
        args.save_steps,
    )
