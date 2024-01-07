import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.multiprocessing import JoinableQueue, Process

import time
from typing import List, Tuple


def naive_process(dataloader: DataLoader, model_blocks: List[Tuple[nn.Module, str]]) -> Tuple[torch.Tensor, float]:
    processed_data = []

    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0]
            for model_block, device in model_blocks:
                batch = model_block(batch.to(device))
            processed_data.append(batch.cpu())
    stop = time.perf_counter()

    processing_time = stop - start
    return torch.concatenate(processed_data, dim=0), processing_time


def model_block_worker(model_block: nn.Module, device: str, input_queue: JoinableQueue, output_queue: JoinableQueue):
    while True:
        batch = input_queue.get()
        if batch is None:
            output_queue.put(None)
            break
        with torch.no_grad():
            processed_batch = model_block(batch.to(device))
        output_queue.put(processed_batch)
        del batch
        input_queue.task_done()
    output_queue.join()


def parallel_process(dataloader: DataLoader, model_blocks: List[nn.Module]) -> Tuple[torch.Tensor, float]:
    processed_data = []
    queues = [JoinableQueue() for _ in range(len(model_blocks) + 1)]
    workers = [
        Process(
            target=model_block_worker,
            args=(model_block, device, queues[i], queues[i + 1]),
            daemon=True,
        )
        for i, (model_block, device) in enumerate(model_blocks)
    ]

    for worker in workers:
        worker.start()

    start = time.perf_counter()
    for batch in dataloader:
        queues[0].put(batch[0])
    queues[0].put(None)

    while True:
        processed_batch = queues[-1].get()
        if processed_batch is None:
            queues[-1].task_done()
            break
        processed_data.append(processed_batch.cpu())
        del processed_batch
        queues[-1].task_done()
    stop = time.perf_counter()

    for worker in workers:
        worker.terminate()

    processing_time = stop - start
    return torch.concatenate(processed_data, dim=0), processing_time