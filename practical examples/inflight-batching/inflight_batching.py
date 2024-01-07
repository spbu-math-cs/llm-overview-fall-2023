import torch
from icecream import ic
from time import time

from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.generation import StoppingCriteriaList


class InflightBatchingCriteria(StoppingCriteria):
    def __init__(self, input_ids, tokenizer, eos_token_id: int, batch_size: int, max_new_tokens: int):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id    # for testing purposes, to ensure unequal generation lengths
        self.batch_size = batch_size

        self.queries_size = input_ids.shape[0]
        self.queries_start_len = input_ids.shape[-1]
        self.queries_ids = input_ids
        self.max_size = input_ids.shape[-1] + max_new_tokens

        st_time = time()
        self.start_time = [st_time]*self.batch_size + [None]*(self.queries_size - self.batch_size)
        self.end_time = [None]*self.queries_size
        self.tokens_generated = 0
        self.outputs = [None]*self.queries_size

        self.order_to_padding = torch.count_nonzero(self.queries_ids == 0, dim=1)
        self.order_to_padding_current = self.order_to_padding[:batch_size]

        # works only in case of batch_size <= self.queries_size
        self.order_to_num = list(range(self.batch_size))
        self.next_num = self.batch_size
        self.worked = self.batch_size


    def __check_stopping(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        # ic(input_ids)
        last_tokens = input_ids[:, -1]
        # ic(last_tokens)

        # ic(input_ids.shape[-1], self.max_size)
        if input_ids.shape[-1] < self.max_size:
            has_no_free_spaces = torch.full((self.batch_size,), False)
            # ic('!!!', has_no_free_spaces)
        else:
            has_no_free_spaces = self.order_to_padding_current == 0
            # ic('???', has_no_free_spaces)

        self.tokens_generated += torch.count_nonzero(input_ids[:, -1] != 0)

        last_token_met = last_tokens == self.eos_token_id
        # ic(last_token_met)

        return has_no_free_spaces | last_token_met


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # ic(input_ids)
        is_done_by_query = self.__check_stopping(input_ids).tolist()
        # ic(is_done_by_query)
        # ic(self.order_to_padding_current)
        for idx, (done, query_num) in enumerate(list(zip(is_done_by_query, self.order_to_num))):
            if not done or self.order_to_num[idx] == self.queries_size:
                continue

            # decode output
            self.outputs[self.order_to_num[idx]] = self.tokenizer.decode(input_ids[idx, :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            self.end_time[self.order_to_num[idx]] = time()
            if self.has_queries_left():
                self.start_time[self.next_num] = time()
            # ic(f'Input {query_num} at index {idx} finished!')

            # check if there are any remaining inputs
            if not self.has_queries_left():
                self.worked -= 1

            # load new input
            if self.has_queries_left():
                input_ids[idx, :] = 0
                # will possibly crash if suddenly receives a query with many tokens
                input_ids[idx, input_ids.shape[-1] - self.queries_start_len:] = self.queries_ids[self.next_num, :]
                self.order_to_padding_current[idx] = input_ids.shape[-1] - self.queries_start_len + self.order_to_padding[self.next_num]
            else:
                input_ids[idx, :] = 0
                self.order_to_padding_current[idx] = self.max_size

            self.order_to_num[idx] = self.next_num
            if self.next_num < self.queries_size:
                self.next_num += 1
            
            # remove paddings if possible
            # ic(self.order_to_padding_current)
            min_padding = torch.min(self.order_to_padding_current)
            # ic(min_padding)
            if min_padding > 0:
                input_ids = torch.Tensor(input_ids[:, min_padding:]).to(torch.int64)
                self.order_to_padding_current -= min_padding
                # ic(input_ids)
                # ic(self.order_to_padding_current)
        
        # ic(self.outputs_ids)
        # ic(f'Working on {self.worked} inputs, next number is {self.next_num}')
        # ic(self.worked == 0)
        return input_ids, self.worked == 0

    def has_queries_left(self):
        return self.next_num < self.queries_size


class StoppingCriteriaMutatorList(StoppingCriteriaList):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        is_done = False
        for criteria in self:
            input_ids, is_done_cur = criteria(input_ids, scores)
            is_done = is_done or is_done_cur
        # ic(is_done)
        return input_ids, is_done
