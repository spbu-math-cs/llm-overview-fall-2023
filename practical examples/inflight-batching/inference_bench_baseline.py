from icecream import ic
from time import time
from statistics import mean, stdev
import torch

from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers.generation import GenerationConfig


BATCH_SIZE = 3
TEST_SIZE = 15
MAX_NEW_TOKENS = 32
EOS_TOKEN_ID = 29889    # replaced eos token because found weights did not generate original eos token often enough

# sample querying
model = LlamaForCausalLM.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    do_sample=True
)

# greedy querying
# model = LlamaForCausalLM.from_pretrained(
#     "NousResearch/Llama-2-7b-hf",
#     do_sample=False,
#     temperature=1,
#     top_p=1
# )

# same inputs
inps = ['Fun fact of the day: ']*TEST_SIZE

# different inputs
# inps = [
#     'Fun fact of the day: ',
#     'The capital city of Germany is ',
#     'Eminem\'s album called ',
#     'Tomorrow the weather is ',
#     'Breaking news: ',
#     'Today we will discuss '
# ]

tokenizer = LlamaTokenizerFast.from_pretrained('NousResearch/Llama-2-7b-hf')
tokenizer.pad_token = '[PAD]'
tokenizer.padding_side = 'left'

gen_cfg = GenerationConfig.from_model_config(model.generation_config)
gen_cfg.max_new_tokens = MAX_NEW_TOKENS
gen_cfg.eos_token_id = EOS_TOKEN_ID
gen_cfg.pad_token_id = 0
ic(gen_cfg)

res_str = list()
st_time, end_time, tok_cnt = list(), list(), list()

for idx in range(0, TEST_SIZE, BATCH_SIZE):
    st_time_val = time()
    st_time.extend([st_time_val]*BATCH_SIZE)

    inp_tok = torch.Tensor(tokenizer(inps[idx:idx+BATCH_SIZE], padding=True)['input_ids']).to(torch.int64)
    inp_tok_start_size = inp_tok.shape[-1]

    res_tok = model.generate(
        inputs=inp_tok,
        generation_config=gen_cfg
    )
    res_str.extend(tokenizer.batch_decode(res_tok, skip_special_tokens=True, clean_up_tokenization_spaces=False))

    end_time_val = time()
    end_time.extend([end_time_val]*BATCH_SIZE)
    tok_cnt.extend((res_tok.shape[-1] - torch.count_nonzero(res_tok[:, inp_tok_start_size:] == 0, dim=1)).tolist())

diff_time = [x - y for (x, y) in zip(end_time, st_time)]

print()
for s in res_str:
    print('...', s)
print(f'Latency:          {mean(diff_time):.4f} ± {stdev(diff_time):.4f} seconds')
print(f'Tokens generated: {sum(tok_cnt)} tokens')
print(f'Throughput:       {sum(diff_time) / sum(tok_cnt):.4f} tokens per second')
print()
