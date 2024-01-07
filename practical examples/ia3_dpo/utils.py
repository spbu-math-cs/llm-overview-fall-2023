from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
from transformers import LlamaForCausalLM

from constants import EPOCH_NUM, SAVE_STEPS
from ia3 import IA3LinearLayer


def train_loop(
    model: LlamaForCausalLM, 
    optimizer: torch.optim.Optimizer, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: Callable,
    checkpoint_prefix: str,
    epoch_num: int = EPOCH_NUM,
    save_steps: int = SAVE_STEPS,
) -> None:
    prev_loss_value = float('inf')
    loss_history = []

    for _ in range(epoch_num):
        for batch in dataloader:
            optimizer.zero_grad()
        
            for key in batch:
                batch[key] = batch[key].to(model.device)
            
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            
            loss_history.append(float(loss))
        
            if len(loss_history) % save_steps == 0:
                avg_loss = np.mean(loss_history[-save_steps:])
                print(f"Avg loss for last {save_steps} steps: {avg_loss}")

                if prev_loss_value > avg_loss:
                    prev_loss_value = avg_loss

                    adapter_weights = {
                        name: OrderedDict(dict(adapter=module.state_dict()["adapter"]))
                        for name, module in model.named_modules() if isinstance(module, IA3LinearLayer)
                    }
                    torch.save(adapter_weights, f"./{checkpoint_prefix}_checkpoint.pth")


def calc_token_log_probs(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    return torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
