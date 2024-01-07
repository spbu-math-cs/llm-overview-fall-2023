## Пререквизиты

```
conda create --name=llama2
conda activate llama2
conda install python=3.8
pip install -r requirements.txt
```

Для запуска бенчмарка без inflight batching вводим `python3 inflight_bench_baseline.py`

Для запуска бенчмарка с использованием inflight batching вводим `python3 inflight_bench.py`

При реализации inflight batching был создан самописный критерий остановки (`StoppingCriteria`), а также модифицированы некоторые функции `GenerationMixin` (`generate`, `sample`)

