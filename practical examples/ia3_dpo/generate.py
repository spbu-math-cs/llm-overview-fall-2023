import argparse

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from constants import INSTRUCTION_PROMPT, MODEL_NAME
from ia3 import enable_adapters, load_adapter_weights, wrap_model


def generate(
    user_input: str,
    checkpoint_path: str = "./dpo_checkpoint.pth",
    model_name: str = MODEL_NAME,
):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(model_name, device_map=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load model
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        low_cpu_mem_usage=True, 
        offload_state_dict=True,
        torch_dtype=torch.float32,
    )

    # wrap model with IA^3 adapters
    wrap_model(model)
    load_adapter_weights(model, checkpoint_path)
    enable_adapters(model)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    prompt = INSTRUCTION_PROMPT.format(user_input)
    input_tokens = tokenizer(prompt, return_tensors="pt")
    for key in input_tokens:
        input_tokens[key] = input_tokens[key].to(device)

    output = model.generate(
        **input_tokens,
        max_new_tokens=30,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(output.cpu().tolist()[0], skip_special_tokens=True)[len(prompt):])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_input",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./dpo_checkpoint.pth",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
    )
   
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        args.user_input,
        args.checkpoint_path,
        args.model_name,
    )
