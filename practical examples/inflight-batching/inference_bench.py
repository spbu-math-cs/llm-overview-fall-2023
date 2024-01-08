from time import time
import torch

from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers.generation import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from modify_generator_mixin import modify_mixin


class EOSCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.enabled = True
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if not self.enabled:
            return False
        return torch.any(input_ids[:, -1] == self.eos_token_id)

    def disable(self):
        self.enabled = False


BATCH_SIZE = 16
TEST_SIZE = 64
MAX_LENGTH = 50
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 29889    # replaced eos token because found weights did not generate original eos token often enough
PAD_TOKEN_ID = 0

# modify_mixin()

model = LlamaForCausalLM.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    do_sample=True
)

inps = ['Fun fact of the day: ']*TEST_SIZE

tokenizer = LlamaTokenizerFast.from_pretrained('NousResearch/Llama-2-7b-hf')
tokenizer.pad_token = '[PAD]'
tokenizer.padding_side = 'left'

res_tok = torch.Tensor(tokenizer(inps[:BATCH_SIZE], padding=True)['input_ids']).to(torch.int64)

gen_cfg = GenerationConfig.from_model_config(model.generation_config)
gen_cfg.max_length = MAX_LENGTH 
gen_cfg.bos_token_id = BOS_TOKEN_ID
gen_cfg.eos_token_id = EOS_TOKEN_ID
gen_cfg.pad_token_id = PAD_TOKEN_ID

# add custom criteria to stop on a single finished sequence
stopping_criteria = StoppingCriteriaList()
custom_criteria = EOSCriteria(
    eos_token_id=EOS_TOKEN_ID
)
stopping_criteria.append(custom_criteria)

st_time = time()
queries_parsed = 0
tokens_generated = 0

order_to_num = list(range(BATCH_SIZE))
used_rows_mask = torch.Tensor([True]*BATCH_SIZE).to(torch.bool)    # used to correctly calculate number of generated tokens
next_num = BATCH_SIZE
outputs_received = 0

while outputs_received < TEST_SIZE:
    start_size = res_tok.shape[-1]

    res_tok = model.generate(
        inputs=res_tok,
        generation_config=gen_cfg,
        stopping_criteria=stopping_criteria
    )

    tokens_generated += torch.count_nonzero(res_tok[used_rows_mask, start_size:] != PAD_TOKEN_ID)

    # find out which sequences are finished
    last_tokens = res_tok[:, -1]
    if res_tok.shape[-1] < MAX_LENGTH:
        has_no_free_spaces = torch.full((BATCH_SIZE,), False)
    else:
        has_no_free_spaces = torch.argmax(torch.where(res_tok != PAD_TOKEN_ID, 1, 0), dim=1) == 0
    last_token_met = (last_tokens == EOS_TOKEN_ID) | (last_tokens == PAD_TOKEN_ID)
    done_queries_idxs = (has_no_free_spaces | last_token_met).nonzero().squeeze(dim=-1).tolist()

    # parse each finished sequence
    for done_idx in done_queries_idxs:
        if order_to_num[done_idx] == TEST_SIZE:
            continue

        # decode and output finished sequence
        s = tokenizer.decode(res_tok[done_idx, :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f'{order_to_num[done_idx]:<3}: {s}')
        order_to_num[done_idx] = next_num

        # fill freed row with empty sequence
        res_tok[done_idx, :] = PAD_TOKEN_ID
        res_tok[done_idx, -1] = BOS_TOKEN_ID

        # fill right part of freed row with new query if possible
        if next_num < TEST_SIZE:
            new_inp_tok = torch.Tensor(tokenizer([inps[next_num]])['input_ids'][0]).to(torch.int64)
            
            start_pos = res_tok.shape[-1] - new_inp_tok.size()[0]
            res_tok[done_idx, start_pos:] = new_inp_tok[:]

            next_num += 1
        else:
            used_rows_mask[done_idx] = False

    outputs_received += len(done_queries_idxs)
    
    padding = torch.argmax(torch.where(res_tok != PAD_TOKEN_ID, 1, 0), dim=1)
    padding = torch.where((res_tok[:, 0] == PAD_TOKEN_ID) & (padding == 0), res_tok.shape[-1], padding)

    min_padding = torch.min(padding)
    if min_padding > 0:
        res_tok = res_tok[:, min_padding:]

end_time = time()
diff_time = end_time - st_time

print()
print(f'>>> Time taken:       {diff_time:.4f} seconds')
print(f'>>> Tokens generated: {tokens_generated} tokens')
print(f'>>> Throughput:       {tokens_generated / diff_time:.4f} tokens per second')
print()
