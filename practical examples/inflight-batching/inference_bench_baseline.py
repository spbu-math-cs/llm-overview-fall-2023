from time import time
import torch

from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers.generation import GenerationConfig

from modify_generator_mixin import modify_mixin


BATCH_SIZE = 16
TEST_SIZE = 64
MAX_LENGTH = 50
EOS_TOKEN_ID = 29889    # replaced eos token because found weights did not generate original eos token often enough
PAD_TOKEN_ID = 0

# modify_mixin()

# sample querying
model = LlamaForCausalLM.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    do_sample=True
)

inps = ['Fun fact of the day: ']*TEST_SIZE

tokenizer = LlamaTokenizerFast.from_pretrained('NousResearch/Llama-2-7b-hf')
tokenizer.pad_token = '[PAD]'
tokenizer.padding_side = 'left'

gen_cfg = GenerationConfig.from_model_config(model.generation_config)
gen_cfg.max_length = MAX_LENGTH
gen_cfg.eos_token_id = EOS_TOKEN_ID
gen_cfg.pad_token_id = PAD_TOKEN_ID

res_str = list()
tokens_generated = 0

st_time = time()

# primitive batch-by-batch parsing
for idx in range(0, TEST_SIZE, BATCH_SIZE):
    inp_tok = torch.Tensor(tokenizer(inps[idx:idx+BATCH_SIZE], padding=True)['input_ids']).to(torch.int64)
    start_size = inp_tok.shape[-1]

    res_tok = model.generate(
        inputs=inp_tok,
        generation_config=gen_cfg
    )
    tokens_generated += torch.count_nonzero(res_tok[:, start_size:] != PAD_TOKEN_ID)

    res_str = tokenizer.batch_decode(res_tok, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for idx_sub, s in enumerate(res_str):
        print(f'{idx + idx_sub:<3}: {s}')

end_time = time()
diff_time = end_time - st_time

print()
print(f'>>> Time taken:       {diff_time:.4f} seconds')
print(f'>>> Tokens generated: {tokens_generated} tokens')
print(f'>>> Throughput:       {tokens_generated / diff_time:.4f} tokens per second')
print()
