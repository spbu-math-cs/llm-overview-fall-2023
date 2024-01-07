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
    DPO_LEARNING_RATE,
    EPOCH_NUM,
    SAVE_STEPS,
    INSTRUCTION_PROMPT,
    BATCH_SIZE,
    DPO_BETA,
)
from ia3 import disable_adapters, enable_adapters, load_adapter_weights, wrap_model
from preprocess_data import PaddingDataCollator, get_texts, replace_letters
from utils import calc_token_log_probs, train_loop


def get_dpo_data(tokenizer: LlamaTokenizer) -> List[str]:
    ru_texts, en_texts = get_texts()
    _, ru_texts_dpo, _, en_texts_dpo = train_test_split(
        ru_texts,
        en_texts,
        train_size=SFT_TRAIN_SIZE_RATIO,
        random_state=RANDOM_STATE,
    )

    texts_dpo = list(
        filter(
            lambda text: len(tokenizer.encode(text)) < TOKENIZED_TEXT_MAX_LEN,
            ru_texts_dpo + en_texts_dpo,
        )
    )

    return texts_dpo


class DPODataset(torch.utils.data.Dataset):
    def __init__(self, data: List[str], prompt: str, tokenizer: LlamaTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt = prompt

        # hardcode only for Llama2
        self.max_length = 4096
        self.max_prompt_length = self.max_length // 2
        
        self.y_w = replace_letters(data)
        self.y_l = data

    def __len__(self) -> int:
        return len(self.y_w)

    def build_tokenized_answer(self, prompt: str, answer: str) -> Dict[str, List[int]]:
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        full_input_ids = np.array(full_tokenized["input_ids"])
    
        # If tokenized prompt is different than both prompt + answer, then it means the last token has changed due to merging
        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length")

        response_token_ids_start_idx = len(prompt_input_ids)

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
    
    def __getitem__(self, index: int) -> Dict[str, torch.LongTensor]:
        output = {}

        y_w = self.y_w[index]
        y_l = self.y_l[index]
        prompt = self.prompt.format(y_l)

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        y_w_tokens = self.build_tokenized_answer(prompt, y_w)
        y_l_tokens = self.build_tokenized_answer(prompt, y_l)
    
        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        y_w_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + y_w_tokens["prompt_input_ids"]
        y_l_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + y_l_tokens["prompt_input_ids"]
    
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        y_w_tokens["prompt_attention_mask"] = [1] + y_w_tokens["prompt_attention_mask"]
        y_l_tokens["prompt_attention_mask"] = [1] + y_l_tokens["prompt_attention_mask"]

        # add EOS token to end of answer
        y_w_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        y_w_tokens["attention_mask"].append(1)
    
        y_l_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        y_l_tokens["attention_mask"].append(1)

        longer_response_length = max(len(y_w_tokens["input_ids"]), len(y_l_tokens["input_ids"]))
    
        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [y_w_tokens, y_l_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
    
        # if that's still too long, truncate the response
        for answer_tokens in [y_w_tokens, y_l_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # create labels for dpo, we only want to predict continuation
        y_w_sequence_tokens = {
            k: y_w_tokens[f"prompt_{k}"] + y_w_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        y_l_sequence_tokens = {
            k: y_l_tokens[f"prompt_{k}"] + y_l_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        
        y_w_sequence_tokens["labels"] = y_w_sequence_tokens["input_ids"][:]
        y_w_sequence_tokens["labels"][:len(y_w_tokens["prompt_input_ids"])] = [0] * len(y_w_tokens["prompt_input_ids"])
        y_l_sequence_tokens["labels"] = y_l_sequence_tokens["input_ids"][:]
        y_l_sequence_tokens["labels"][:len(y_l_tokens["prompt_input_ids"])] = [0] * len(y_l_tokens["prompt_input_ids"])
    
        for k, toks in {
            "y_w_": y_w_sequence_tokens,
            "y_l_": y_l_sequence_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                output[f"{k}{type_key}"] = tokens

        return output


def dpo_loss(
    pi_model: LlamaForCausalLM,
    batch: Dict[str, torch.LongTensor],
    beta: float = DPO_BETA,
) -> torch.FloatTensor:
    # calc ref policy logits
    pi_model.eval()
    disable_adapters(pi_model)
    with torch.no_grad():
        ref_model_y_w_logits = pi_model(
            batch["y_w_input_ids"],
            batch["y_w_attention_mask"],
        ).logits[:, :-1, :]
        
        ref_model_y_l_logits = pi_model(
            batch["y_l_input_ids"],
            batch["y_l_attention_mask"],
        ).logits[:, :-1, :]

    # calc current policy logits
    pi_model.train()
    enable_adapters(pi_model)
    
    pi_model_y_w_logits = pi_model(
        batch["y_w_input_ids"],
        batch["y_w_attention_mask"],
    ).logits[:, :-1, :]

    pi_model_y_l_logits = pi_model(
        batch["y_l_input_ids"],
        batch["y_l_attention_mask"],
    ).logits[:, :-1, :]

    y_w_labels = batch["y_w_labels"][:, 1:]
    y_w_mask = y_w_labels != 0
    y_l_labels = batch["y_l_labels"][:, 1:]
    y_l_mask = y_l_labels != 0
    
    pi_model_y_w_logprobs = (calc_token_log_probs(pi_model_y_w_logits, y_w_labels) * y_w_mask).sum(1)
    pi_model_y_l_logprobs = (calc_token_log_probs(pi_model_y_l_logits, y_l_labels) * y_l_mask).sum(1)
    ref_model_y_w_logprobs = (calc_token_log_probs(ref_model_y_w_logits, y_w_labels) * y_w_mask).sum(1)
    ref_model_y_l_logprobs = (calc_token_log_probs(ref_model_y_l_logits, y_l_labels) * y_l_mask).sum(1)

    pi_logratios = pi_model_y_w_logprobs - pi_model_y_l_logprobs
    ref_logratios = ref_model_y_w_logprobs - ref_model_y_l_logprobs
    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
    
    return loss


def dpo_train(
    model_name: str = MODEL_NAME,
    sft_model_checkpoint_path: str = "./sft_checkpoint.pth",
    batch_size: int = BATCH_SIZE,
    learning_rate: float = DPO_LEARNING_RATE,
    epoch_num: int = EPOCH_NUM,
    save_steps: int = SAVE_STEPS,
):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(model_name, device_map=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # make dataloader
    dpo_data = get_dpo_data(tokenizer)
    dpo_train_dataset = DPODataset(
        dpo_data,
        INSTRUCTION_PROMPT,
        tokenizer,
    )
    dpo_dataloader = torch.utils.data.DataLoader(
        dpo_train_dataset,
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
    load_adapter_weights(model, sft_model_checkpoint_path)

    dpo_optimizer = torch.optim.RMSprop(
        [param for param in model.parameters() if param.requires_grad], 
        lr=learning_rate,
    )

    train_loop(
        model, 
        dpo_optimizer, 
        dpo_dataloader, 
        dpo_loss,
        "dpo",
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
        "--sft_model_checkpoint_path",
        type=str,
        default="./sft_checkpoint.pth",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=DPO_LEARNING_RATE,
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
    dpo_train(
        args.model_name,
        args.sft_model_checkpoint_path,
        args.batch_size,
        args.learning_rate,
        args.epoch_num,
        args.save_steps,
    )
