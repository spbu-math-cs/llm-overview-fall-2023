# Сравнение реализаций Pipeline Parallelism (автор: Влад Ушаков)

В Jupyter ноутбуке представлено сравнение двух реализаций Pipeline Parallelism на Python + PyTorch. Одна из них, названная **наивной**, выполняет обработку батчей данных последовательно. Другая, названная **параллельной**, использует одновременно несколько устройств при обработке последовательно приходящих данных.