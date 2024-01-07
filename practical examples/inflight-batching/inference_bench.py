from icecream import ic
from statistics import mean, stdev
import torch

from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers.generation import GenerationConfig

from inflight_batching import InflightBatchingCriteria, StoppingCriteriaMutatorList
from modify_generator_mixin import modify_mixin


BATCH_SIZE = 3
TEST_SIZE = 15
MAX_NEW_TOKENS = 32
EOS_TOKEN_ID = 29889    # replaced eos token because found weights did not generate original eos token often enough

modify_mixin()

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

inp_tok = torch.Tensor(tokenizer(inps, padding=True)['input_ids']).to(torch.int64)
inp_tok_for_model = inp_tok[:BATCH_SIZE, :]

gen_cfg = GenerationConfig.from_model_config(model.generation_config)
gen_cfg.max_new_tokens = MAX_NEW_TOKENS
gen_cfg.eos_token_id = EOS_TOKEN_ID
gen_cfg.pad_token_id = 0
ic(gen_cfg)

stopping_criteria = StoppingCriteriaMutatorList()
stopping_criteria.append(InflightBatchingCriteria(
    input_ids=inp_tok,
    tokenizer=tokenizer,
    eos_token_id=EOS_TOKEN_ID,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS
))

_ = model.generate(
    inputs=inp_tok_for_model,
    generation_config=gen_cfg,
    stopping_criteria=stopping_criteria
)

res_str = stopping_criteria[0].outputs
diff_time = [x - y for (x, y) in zip(stopping_criteria[0].end_time, stopping_criteria[0].start_time)]

print()
for s in res_str:
    print('...', s)
print(f'Latency:          {mean(diff_time):.4f} ± {stdev(diff_time):.4f} seconds')
print(f'Tokens generated: {stopping_criteria[0].tokens_generated} tokens')
print(f'Throughput:       {stopping_criteria[0].tokens_generated / sum(diff_time):.4f} tokens per second')
print()
