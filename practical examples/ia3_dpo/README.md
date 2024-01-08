# Описание

Задача: реализовать $\text{IA}^{3}$[^1] и DPO[^2] для Llama2-7b[^3], чтобы в своих ответах модель заменяла английскую букву c на русскую, английскую y на русскую, английскую o на русскую и наоборот.

## Установка зависимостей
```bash
conda create --name=ia3_dpo
conda activate ia3_dpo
conda install python=3.10.12
pip install -r requirements.txt
```

## Обучение SFT модели для DPO

Обучение ведётся на части следующего [датасета](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7/viewer/default/ru_mnli). Была использовалась смесь русских текстов (из колонки "premise") и английских текстов (из колонки "premise original").

Обучаются только $\text{IA}^{3}$-адаптеры, которые сохраняются в корне директории по пути "./sft_checkpoint.pth"

Обучение запускается следующей командой:

```bash
export PYTHONPATH=$PWD &&  python sft.py --model_name <название модели из Hugging Face Hub, по умолчанию NousResearch/Llama-2-7b-hf> --batch_size <размер батча для обучения, по умолчанию 16> --learning_rate <learning rate, по умолчанию 1e-4> --epoch_num <количество эпох для дообучения, по умолчанию 5> --save_steps <количество шагов, когда происходит сейв весов модели, если среднее значение лосса уменьшилось>
```
Более простая команда:
```bash
export PYTHONPATH=$PWD &&  python sft.py
```

## Обучение DPO модели

Обучение ведётся на оставшейся части упомянутого ранее [датасета](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7/viewer/default/ru_mnli). Была использовалась смесь русских текстов (из колонки "premise") и английских текстов (из колонки "premise original"), тексты ни на одном из языков не пересекаются с теми, что были использованы при обучении SFT модели.

Обучаются только $\text{IA}^{3}$-адаптеры, которые сохраняются в корне директории по пути "./dpo_checkpoint.pth"

Обучение запускается следующей командой:

```bash
export PYTHONPATH=$PWD &&  python dpo.py --model_name <название модели из Hugging Face Hub, по умолчанию NousResearch/Llama-2-7b-hf> --sft_model_checkpoint_path <веса адаптеров, которые будут использованы в качестве инициализации для DPO модели и референсной модели, участвующей в функции потерь, по умолчанию ./sft_checkpoint.pth> --batch_size <размер батча для обучения, по умолчанию 16> --learning_rate <learning rate, по умолчанию 1e-4> --epoch_num <количество эпох для обучения, по умолчанию 5> --save_steps <количество шагов, когда происходит сейв весов модели, если среднее значение лосса уменьшилось>
```

Более простая команда:
```bash
export PYTHONPATH=$PWD &&  python dpo.py
```

## Генерация при помощи DPO модели

Для генерации с помощью DPO модели используется следующая команда:

```bash
export PYTHONPATH=$PWD &&  python generate.py --user_input "сообщение пользователя, обязательно в кавычках!" --checkpoint_path <веса адаптеров, которые будут использованы для генерации, по умолчанию ./dpo_checkpoint.pth> --model_name <название модели из Hugging Face Hub, по умолчанию NousResearch/Llama-2-7b-hf>
```

Более простая команда:
```bash
export PYTHONPATH=$PWD &&  python generate.py --user_input "сообщение пользователя, обязательно в кавычках!"
```

Ответ выведется в консоли, после чего программа завершится.

## Результаты обучения

Ниже дана сравнительная таблица порождения для разных моделей (`do_sample=True`, `max_new_tokens=100`), выделенные жирным шрифтом буквы отображают факт замены буквы на соотвествующую из другого языка:

| Оригинальное предложение  | Модель до дообучения | SFT модель| DPO модель| 
|-|-|-|-|
| Хитрую сороку поймать морока, А сорок сорок — сорок морок.  | Хитрую сороку поймать морока, А сорок сорок — сорок морок.\n\n## Hint\n\nThe first letter of the answer is the first letter of the text.\n\n## Solution\n\nThe answer is:\n\nХитрую сороку поймать морока, А сорок сорок — сорок морок. | Һитрую **c**орокy поймать морока, А **c**орок **c**орок — **c**орок морок. | Хытр**y**у с**o**р**o**к**y** п**o**ймать м**o**рокa, А **co**р**o**к**c** **co**р**o**к**c** — **co**р**o**к**c** м**o**рокa.|
| Осип охрип, а Архип осип. Охрип Осип, а осип Архип. Осип Архип, а Осип охрип. Архип охрип, а Осип осип.  | Осип охрип, а Архип осип. Охрип Осип, а осип Архип. Осип Архип, а Осип охрип. Архип охрип, а Осип осип.\n\n## See also\n\n* Russian orthography\n* Russian phonology\n* Russian phonotactics\n* Russian phonemics\n* Russian phonology | О**c**ип охрип, а Архип о**c**ип. Охрип О**c**ип, а о**c**ип Архип. О**c**ип Архип, а О**c**ип охрип. Архип охрип, а О**c**ип о**c**ип.| О**c**п **o**хрип, а Архип **oc**п. Охрип О**c**п, а **oc**п Архип. О**c**п Архип, а О**c**п **o**хрип. Архип **o**хрип, а О**c**п **oc**п.| 
| You’re yacking and yawning and unusually annoying | \\begin{code}\nYou’re yacking and yawning and unusually annoying\n\\end{code} | ыacking and **у**awning and unusuall**у** anno**у**ing| Юa**с**king and **у**awning and unusuall**у** anno**у**ing |
| The coat from the coast cost more than the most.| The coat from the coast cost more than the most. | The c**о**at fr**о**m the c**о**ast c**о**st m**о**re than the m**о**st. | The **со**at fr**о**m the **со**ast **со**st m**о**re than the m**о**st. |
| Из жуткого гнезда я вижу весь мир от края до края, и потому со своей высоты без малейшей тени сомнений провозглашаю, дело в шляпе! | Из жуткого гнезда я вижу весь мир от края до края, и потому со своей высоты без малейшей тени сомнений провозглашаю, дело в шляпе! | Ыз ж**y**тк**o**г**o** гнезда я виж**y** весь мир от края до края, и п**o**т**o**м**y** с**o** своей высоты бeз малейш**o**й тeни сомнeний п**o**зв**o**згaю, дело в шляпе! | Зж**y**тк**o**г**o** гнезда я выж**y** весь мр **o**т края д**o** края, и п**o**т**o**м**y** с**o** св**o**ей выс**o**ты бeз малейш**o**й тни с**o**мнeний п**o**зв**o**зглaз**y**, дел**o** в шляпе! |

[^1]: Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning: https://arxiv.org/abs/2205.05638
[^2]: Direct Preference Optimization: Your Language Model is Secretly a Reward Model: https://arxiv.org/abs/2305.18290
[^3]: Для этого была взяты веса следующей модели: https://huggingface.co/NousResearch/Llama-2-7b-hf
