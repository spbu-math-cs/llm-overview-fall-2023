# Оценка качества

# Введение

- Точечные оценки
- Графические методы
- Энтропия
- Перплексия

В каждой задаче машинного обучения важна оценка результатов моделей. Более того, важно учесть правильный выбор метрик для поставленной задачи.

Для досконального и качественного изучения оценок качества рассмотрим, прежде всего, бинарную классификацию.

Для классификаций данного рода имеется 4 вида исходов: истинно положительные (TP), истинно отрицательные (TN), ложно положительные (FP) и ложно отрицательные (FN). В таких случаях можно применять следующие метрики:

*в форме точечных оценок*:

1. ****Accuracy**** (Точность), которая показывает долю правильно проставленных меток класса от общего количества данных:

**Accuracy** = (TP + TN)/(TP + TN + FP + FN) 

Рассмотрим пример: мы хотим оценить работу спам-фильтра электронной почты. У нас есть 100 не-спам писем, 90 из которых наш классификатор определил верно (то есть TN = 90, FP = 10), и 10 спам-писем, 5 из которых классификатор также определил верно (TP = 5, FN = 5).

В таком случае, Accuracy примет значение:

**Accuracy** = (TP + TN)/(TP + TN + FP + FN) = (5 + 90)/(5 + 90 + 10 + 5) ~ 0.864.

Но если мы просто будем предсказывать все письма как не-спам, то получим более высокий результат:

**Accuracy** = (TP + TN)/(TP + TN + FP + FN) = (0 + 100)/(0 + 100 + 0 + 10) ~ 0.909.

Более того, наша модель совершенно не обладает никакой предсказательной силой, так как изначально мы хотели определять письма со спамом.

1. ****Precision**** (как ни странно, тоже Точность), которая показывает долю истинно положительных исходов от всего набора положительных меток:

****Precision**** = ********TP / (TP + FP)

В случае с примером по спам-фильтру, ****Precision**** = ********TP / (TP + FP) = 5 / (5 + 10) ~ 0.333

1. ****Recall**** (Чувствительность), которая показывает долю положительных среди всех меток класса, которые были определены как «положительный».

****Recall =**** TP / (TP + FN)

В случае с примером по спам-фильтру, ****Recall =**** TP / (TP + FN) = 5 / (5 + 5) = 0.5

1. ****F_1-Score**** (применяется, если Precision и Recall совпадают), которая вычисляет их среднее гармоническое для получения оценки результатов:

****F_1-Score**** = ****(Precision * Recall)**** / (****Precision**** + ****Recall).****

В случае с примером по спам-фильтру, ****F_1-Score**** = ****(Precision * Recall)**** / (****Precision**** + ****Recall)**** = (0.333 * 0.5)/(0.333 + 0.5) ~ 0.2

С точки зрения математической статистики, FP соответствует количеству ошибок I-го рода, а FN - количеству ошибок II-го рода. Более того, FP (как аналог ошибки I-го рода) описывает слабость (иными словами, ***немощность***) критерия, по которому осуществляется бинарная классификация, в то время как FN - слепоту (иными словами, ***нечувствительность***) критерия.

И в случае несбалансированных выборок в метриках по FP и FN будут происходить существенные искажения.

*в форме графических методов*:

Для данных метрик введём вспомогательные переменные - True Positive Rate (TPR) и False Positive Rate (FPR):

TPR = TP /(TP + FN)

FPR = FP /(FP + TN)

1. **ROC**, показывающий зависимость верно классифицируемых объектов положительного класса (TPR) от ложно положительно классифицируемых объектов негативного класса (FPR).
2. ****AUC****, вычисляющая площадь под кривой ROC.

Рассмотрим следующую задачу:

нам необходимо выбрать 100 релевантных документов из 1 миллиона документов. Мы обучили два алгоритма:

**Алгоритм 1** возвращает 100 документов, 90 из которых релевантные.

То есть TPR = 0.9, FPR ~ 0.00001.

**Алгоритм 2** возвращает 2000 документов, 90 из которых релевантные.

То есть TPR = 0.9, FPR ~ 0.00191.

На первый взгляд кажется, что первый алгоритм лучше, т.к. FPR меньше. Однако разница в False Positive Rate между этими двумя алгоритмами очень мала — только 0.0019. Причиной тому является AUC-ROC, который измеряет долю FP относительно TN. И в задачах, где нам не так важен больший класс, может делать некорректные выводы при сравнении алгоритмов.

Поэтому вернёмся к ****Precision**** и ****Recall****:

**Алгоритм 1:**

****Precision**** = 0.9 , ****Recall**** = 0.9

**Алгоритм 2:**

****Precision**** = 0.045, ****Recall**** = 0.9

И именно здесь заметна значительная разница в точности между двумя алгоритмами — 0.855.

Также не стоит забывать и мульти-классификацию, где происходит вычисление среднего метрики по всем классам. Тогда в качестве **«положительного»** класса берется вычисляемый, а все остальные — в качестве **«отрицательного»**.

Помимо метрик стоит упомянуть и такую характеристику как **энтропия,** которая фактически является мерой беспорядка. Чем легче будет распределить объекты на классы, тем ниже энтропия.

Также стоит упомянуть и другие описания энтропии - данная величина является показателем того, как сильно мы можем ***компактизировать*** битовую информацию о сообщении. Говоря более простыми словами, энтропия ограничивает максимально возможное сжатие без потерь (или почти без потерь), которое может быть реализовано при использовании теоретически (то есть, чем проще упростить информацию, тем ниже энтропия).

По формуле Шеннона, энтропия представляется следующим образом:

$H(x) = - sum_i (p_i * log p_i)$

Скобка с логарифмом обозначает измеряемое в битах количество информации, содержащейся в том событии, что случайная величина приняла значение. В то время как H(x) - количество информации, которое в среднем приходится на одно событие.

У энтропии есть значимые преимущества:

1. В отличие от энтропии, метрика Accuracy очень чувствительна к порядку данных в тренировочном наборе, а также не учитывается статистическая достоверность.
2. Энтропия хорошо сочетается с сигмоидой и моделью логистической регрессией, которая занимается распределениями, а не списком вопросов.

Ещё одним важным инструментом в оценке качества моделей машиной обучения является **перплексия** - мера того, насколько хорошо распределение вероятностей предсказывает выборку.

Выражается перплексия следующим образом:

$PP(p) = 2^H, H = H(p)$

Основание логарифма не обязательно должно быть равно 2: перплексия не зависит от основания логарифма при условии, что энтропия и показательная функция имеют одно и то же основание. 

# Оценки без привлечения людей

Развитие методов машинного перевода потребовало создание метрик, более адекватно оценивающих его качество. Проблема перплексии следующая - это внутренняя оценка модели, оценивающая уверенность модели в предсказанном результате, однако, качество самого результата никоим образом не оценивается - два грамматически похожих предложения, имеющих противоположный семантический смысл, могут получить одинаковую оценку перпелксии.

Кроме того, перплексия также имеет [свои недостатки](https://arxiv.org/abs/2106.00085) при оценке качества моделей (анализ приведен в [статье](https://arxiv.org/abs/2210.05892)):

1. Значение перплексии для коротких текстов склонно быть больше, чем у длинных текстов.
2. Повторяющиеся фрагменты текста приводят к более низким значениям перплексии, несмотря на то, что повторы часто не улучшают качество текста.
3. Значения перплексии могут существенно зависеть от наличия и использования знаков пунктуации в тексте.

## **BLEU**

*Основана на подсчете слов (unigrams) и словосочетаний (n‑grams) из машинного текста, также встречающихся в эталоне. Далее это число делится на общее число слов и словосочетаний в машинном переводе — получается precision. К итоговому precision применяется корректировка — штраф за краткость (brevity penalty), чтобы избежать слишком высоких оценок BLEU для кратких и неполных переводов*

### Вычисление:

- Дано:
    - Текст, качество которого необходимо измерить
    - Эталонные переводы
- Алгоритм:
    1. Считаются отношения количества совпадающих n-грамм к их количеству в измеряемом тексте
    2. Считается штраф за краткость BP (brevity penalty)
    3. Значение метрики вычисляется следующим образом для каждого из эталонных примеров:
    4. Берется взвешенное среднее значений метрик из пункта 3
    

$$
\text{MP}_n = \frac{\text{Number of n-gram matches}}{\text{Total number of n-grams in candidate translation}}

$$

MP(modified precision) - количество совпадений n-грамм ограничено количеством оных в эталонном предложении

$$
\text{BP} = \begin{cases} 
1 & \text{if |candidate|} > \text{|reference|} \\
\exp\left(1 - \frac{{\text{reference length}}}{{\text{candidate length}}}\right) & \text{if |candidate|} \leq \text{|reference|}
\end{cases}
$$

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\frac{1}{N} \sum_{i=1}^{N} \text{MP}_i\right)
$$

Также существуют отдельно BLEU_n, где 1 ≤ n ≤ 4, в которых для сравнения используются, соотвественно, n-граммы. Это необходимо, так как, например, при использовании BLEU_1 от изменения порядка слов результат не изменится (например, “Он был не зеленого, а красного цвета” и “Он был зеленого, а не красного цвета”), чего нельзя сказать о значении самого предложения.

Плюсы

- Простота интерпретации
- Распространенность

Минусы

- Учитывает исключительно precision
- Не учитывает семантической близости слов

## ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) - набор метрик в области обработки, используемых для оценки качества **суммаризации** и *машинного перевода* естественного языка. Метрики сравнивают автоматически созданное краткое изложение или перевод с эталонным или набором эталонов (созданных человеком) краткого изложения или перевода
Строго говоря, та же BLEU без штрафа за краткость BP, в которой вместо precision используется recall

### ROUGE-1

$$
ROUGE-1_{recall} = \frac{\text{Num word matches}}{\text{Num words in reference}}
$$

$$
ROUGE-1_{precision} = \frac{\text{Num word matches}}{\text{Num words in generated summary}}
$$

### ROUGE-2

$$
ROUGE-2_{recall} = \frac{\text{Num bigram matches}}{\text{Num bigrams in reference}}
$$

$$
ROUGE-2_{precision} = \frac{\text{Num bigram matches}}{\text{Num bigrams in generated summary}}
$$

### ROUGE-L

Здесь LCS(gen, ref) - длина наибольшей совпадающей упорядоченной подпоследовательности 

$$
ROUGE-L_{recall} = \frac{\text{LCS(gen, ref)}}{\text{Num words in reference}}
$$

$$
ROUGE-L_{precision} = \frac{\text{LCS(gen, ref)}}{\text{Num words in generated summary}}
$$

### ROUGE (F1-score)

Чтобы учитывать в оценку длину генерируемого текста, для вышеперечисленных считают F1-score:

$$
ROUGE_{F_1} = 2\cdot \frac{ROUGE_{precision}\cdot ROUGE_{recall}}{ROUGE_{precision}+ ROUGE_{recall}}
$$

Плюсы

- Простота и понятность
- Распространенность
- Использование recall и F1

Минусы

- Не учитывает семантической близости слов

## METEOR

Строится на подсчете соответствий отдельных слов *(униграмм)*, включая вычисление precision, recall и F1 показателя гармонического среднего. При этом учитываются не только точные совпадения слов (как в BLEU или ROUGE), но и присутствие однокоренных слов или синонимов.

## Формула

$$
F_{mean}= \frac{10PR}{R+9P}
$$

, где P и R - precision и recall для униграмм

$$
p = 0.5 \left(\frac{c}{u_m} \right)^3
$$

- штраф за фрагментацию, где с - количество фрагментов (совпадающих подпоследовательностей длины ≥2), a u_m - длина обрабатываемой строки.

$$
METEOR = p \cdot F_{mean}
$$

Плюсы

- Использование синонимов
- Использование recall и F1

Минусы

- Неоднозначность в оценке фрагментации (структуры предложений могут сильно отличаться)
- Более трудоемкое вычисление

## Корреляция с человеческими оценками

Проблема в том, что представленные выше метрики *очень слабо* коррелируют с человеческими оценками, что было показано в [указанной статье](https://arxiv.org/abs/1603.08023)

![Снимок экрана 2023-12-10 в 23.32.46 (1).png](assets/1.png)

# Нейросетевые оценки без привлечения людей

Большинство метрик, предложенных после 2016 года – нейросетевые. Первым шагом в применении нейросетей в расчете метрик стало использование векторных представлений слов (embeddings).

> **Эмбеддинг** (embedding) – это векторное представление слова, то есть кодировка слова через набор чисел, получаемая на выходе из специальных моделей, анализирующих словоупотребление в больших наборах текстов.
> 
- Сначала близость embeddings машинного и эталонного переводов оценивалась эвристическими методами (например, для метрик WMD, BERTScore, YiSi).
- Далее появились нейросети, последние слои которых принимают на вход embeddings машинного и эталонного переводов, а на выходе дают оценку качества перевода (такие как BLEURT, Prism).
- Затем возникли модели, на вход которых, помимо машинного и эталонного перевода, также может подаваться первоисточник – оригинал переводимого текста (COMET, UniTE).
- В параллель в рамках решения задачи Quality Estimation (см. главу про безрефенесные оценки) развивались модели, сравнивающие напрямую машинный перевод и первоисточник, без эталонного перевода. Так появилось то, что можно назвать безреференсными метриками (reference-free metrics).

Развитие всех этих методов происходит для того, чтобы подойти к созданию такой метрики, степень корреляции которой с человеческой оценкой будет наименьшей.

Подход к созданию эмбеддингов был представлен Mikolov et al. в 2013 в работе [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf). Алгоритм получения эмбеддингов, представленный Mikolov et al., известен как  Word2Vec. 
Впоследствии был создан ряд других алгортимов, например: GloVe, fastText, doc2vec, ELMo, BERT.

![https://sslprod.oss-cn-shanghai.aliyuncs.com/stable/slides/Unsupervised_Learning_Word_Embedding/Unsupervised_Learning_Word_Embedding_1440-03.jpg](https://sslprod.oss-cn-shanghai.aliyuncs.com/stable/slides/Unsupervised_Learning_Word_Embedding/Unsupervised_Learning_Word_Embedding_1440-03.jpg)

## **Референсные оценки**

После появления эмбеддингов стали возникать метрики, которые оценивают уже не лексическое, а семантическое сходство машинного перевода с эталонным (грубо говоря, не совпадение слов, а совпадение смыслов).

Все метрики, сравнивающие эмбеддинги, можно считать нейросетевыми, поскольку сами эмбеддинги получают в результате обучения различных моделей.

Рассмотрим наиболее известные из референсных нейросетевых метрик.

### **ReVal**

[ReVal (Gupta et al., 2015)](https://aclanthology.org/D15-1124.pdf) считается первой нейросетевой метрикой, предложенной непосредственно для оценки качества машинного перевода.

Вычисление данной метрики выполняется с использованием рекуррентной  (отсюда и название метрики) нейросетевой модели LSTM, а также вектора слов GloVe.

ReVal существенно лучше коррелирует с человеческими оценками качества перевода, чем традиционные метрики, но хуже, чем более поздние нейросетевые метрики.

### **BERTScore**

BERTScore – метрика, предложенная [Zhang et al. в 2019](https://arxiv.org/abs/1904.09675) для оценки качества генерируемого текста. Основана на оценке близости контекстных эмбеддингов, полученных из предобученной нейросетевой модели BERT.

Для расчета BERTScore близость двух предложений – сгенерированного моделью и эталонного – оценивается как сумма косинусных подобий между эмбеддингами слов, составляющих эти предложения.

![https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EEVR8WZHAjXkt7WrlsGtYQ.png](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EEVR8WZHAjXkt7WrlsGtYQ.png)

![Screenshot 2023-12-10 at 17.56.40.png](assets/Screenshot_2023-12-10_at_17.56.40.png)

### **BLEURT**

BLEURT (Bilingual Evaluation Understudy with Representations from Transformers) – еще одна метрика на базе embeddings из BERT, предложенная [Sellam et al. в 2020](https://arxiv.org/abs/2004.04696) для оценки качества генерации текста.

![Screenshot 2023-12-10 at 17.57.57.png](assets/Screenshot_2023-12-10_at_17.57.57.png)

Расчет метрики BLEURT выполняется нейросетевыми методами. Для целей расчета BLEURT модель BERT была дообучена на:

- cинтетическом датасете в виде пар предложений из Wikipedia;
- открытом наборе переводов и присвоенных им человеком рейтингов из WMT Metrics Shared Task.

BLEURT обучается на открытом наборе рейтингов, — WMT Metrics Shared Task. При этом пользователь может добавить дополнительные рейтинги для обучения. Пайплайн обучения BLEURT предполагает:

- Использование контекстных представлений слов из BERT;
- Новая схема предобучения для увеличения устойчивости метрики
- Схема обучения BERT для BLEURT на картинке.

Ранее векторные представления из BERT уже использовались для оценки моделей: YiSi или BERTscore. BLEURT предобучается дважды. Сначала с использованием целевой функции языковой модели. Затем модель дообучается на датасете WMT Metrics, на наборе рейтингов пользователи или на комбинации этих двух наборов данных.

### **Prism**

Prism (Probability is the metric) – метрика качества машинного перевода, предложенная [Thompson, Post в 2020](https://arxiv.org/abs/2212.02988) на базе их собственной мультиязычной модели-трансформера Prism.

![Screenshot 2023-12-10 at 18.01.57.png](assets/Screenshot_2023-12-10_at_18.01.57.png)

Авторы отметили сходство задачи оценки близости машинного перевода к эталонному и задачи оценки вероятности парафраза. В качестве оценки принята вероятность такого парафраза, при котором эталонному переводу (выполненному человеком) соответствовал бы текст машинного перевода.

Учитывая данный подход, для обучения модели не потребовалось использовать человеческие оценки качества перевода.

### **COMET**

COMET (Crosslingual Optimized Metric for Evaluation of Translation) – метрика, предложенная [Rei et al. в 2020](https://aclanthology.org/2020.emnlp-main.213.pdf) на базе собственной модели и целого фреймворка для тренировки других моделей оценки качества перевода.

COMET использует в качестве энкодера мультиязычную модель XLM-RoBERTa, поверх которой добавлены дополнительные слои, на выходе которых – оценка качества перевода. Модель принимает на вход не только машинный перевод (hypothesis) и эталон (reference), но и переводимый текст-первоисточник (source).

Модель обучена на триплетах данных hypothesis-source-reference, а также рейтингах перевода, данные человеком (из WMT, как и для BLEURT). Обучение выполнено путем минимизации среднеквадратического отклонения (Mean Squared Loss, MSE) оценок, данных моделью, от истинных рейтингов перевода.

![Screenshot 2023-12-10 at 19.19.46.png](assets/Screenshot_2023-12-10_at_19.19.46.png)

Таким образом, мы рассмотрели наиболее известные референсные нейросетевые метрики. Они, как правило, лучше коррелируют с человеческой оценкой качества перевода, чем традиционные, однако имеют и свои недостатки. В первую очередь, это необъяснимость тех или иных оценок, поскольку нейросетевые расчеты выполняется по принципу «черного ящика». Также отметим более высокие, по сравнению с традиционными метриками, требования к вычислительным ресурсам.

## **Безреференсные системы оценки**

В NLP cуществует отдельная задача Quality Estimation (QE) – предсказание качества машинного перевода в отсутствие референса, то есть без ориентира в виде эталонного перевода, выполненного человеком. Задача QE может решаться как на уровне отдельных слов, так и на уровне предложений.

[Современный подход](https://aclanthology.org/2022.findings-acl.327.pdf) к QE заключается в дообучении мощных предобученных многоязычных нейросетевых моделей-энкодеров (таких как BERT, XLM-R) для прямого сравнения машинного перевода и первоисточника на исходном языке.

В последние годы модели QE достигли довольно высокого уровня корреляции с человеческой оценкой качества перевода предложений (коэффициент корреляции Пирсена до 0.9 для некоторых языковых пар - [WMT2020: Specia et al., 2020](https://www.statmt.org/wmt20/pdf/2020.wmt-1.79.pdf)).

Оценки, получаемые на выходе моделей QE, по сути представляют собой безреференсные метрики качества машинного перевода. Их можно рассматривать как альтернативу значительно более распространенным традиционным и референсным нейросетевым метрикам.

Обучение моделей QE выполняется на датасетах, содержащих человеческие оценки качества перевода. Также выполненные человеком оценки используются и при определении степени адекватности метрик качества машинного перевода. 

Рассмотрим подробней применение данного подхода для построения некоторых оценок качества моделей.

### COMET-Kiwi

Модель COMET-Kiwi ([Rei et al., 2022](https://www.inesc-id.pt/publications/18933/pdf)) агрегирует две QE модели. Первая из них имеет классическую для QE моделей архитектуру «предсказатель-оценщик», в рамках которой выполняется совместный энкодинг машинного перевода и первоисточника. Эта модель обучена на выполненных человеком оценках качества машинного перевода (а именно Direct Assessment, данные WMT за период 2017-2019) и дообучена на датасете MLQE-PE ([Multilingual Quality Estimation and Post-Editing Dataset, 2021](https://arxiv.org/pdf/2010.04480.pdf)).

Вторая модель – та же модель разметки последовательностей, обученная на данных MQM, что и в COMET версии 2022 года (COMET-22, [Rei et al., 2022](https://aclanthology.org/2022.wmt-1.52.pdf)), но не использующая референс.

Признаки, извлеченные из обеих QE моделей, взвешиваются при помощи настраиваемых гиперпараметров таким образом, чтобы на выходе получить одну агрегированную оценку.

## REUSE

Метрика REUSE ([REference-free UnSupervised quality Estimation Metric, Mukherjee and Shrivastava, 2022](https://www.researchgate.net/publication/366398782_REUSE_REference-free_UnSupervised_quality_Estimation_Metric)). Для расчета этой метрики переведенное и исходное предложения дополнительно разбиваются на словосочетания. Оценка качества перевода выполняется как на уровне предложений, так и на уровне словосочетаний.

Близость словосочетаний оценивается на основе контекстных эмбеддингов из модели BERT, а близость предложений – на основе эмбеддингов модели LaBSE (Language-agnostic BERT Sentence Embeddings, [Feng et al., 2020](https://arxiv.org/abs/2007.01852)). Далее выполняется усреднение оценок степени близости по всем словосочетаниям и предложениям, что и дает итоговое значение метрики REUSE.

![Screenshot 2023-12-10 at 22.41.10.png](assets/Screenshot_2023-12-10_at_22.41.10.png)

## HWTSC-Teacher-Sim

Метрика HWTSC-Teacher-Sim предложена Huawei и описана в [Liu et al., 2022](https://aclanthology.org/2022.wmt-1.48.pdf) (первоисточник [Zhang et al., 2022b](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_12) – за пэйволлом). Для расчета этой метрики применяется мультиязыковая дистилляция знаний ([Reimers and Gurevych, 2020b](https://aclanthology.org/2020.emnlp-main.365.pdf)), путем которой обеспечивается соответствие эмбеддингов языков перевода и оригинала – см. схему ниже.

В качестве модели-учителя использована моноязычная Sentence BERT ([Reimers and Gurevych, 2019](https://aclanthology.org/D19-1410.pdf)). В качестве модели-ученика могут использоваться многоязычные модели (например, mBERT либо XLM-R). Дистилляция позволяет максимизировать близость пар предложений на языках оригинала и перевода. После дистилляции выполняется дообучение модели на данных MQM, после чего оценка степени близости, даваемая этой моделью, может быть использована в качестве метрики качества перевода.

![Screenshot 2023-12-10 at 22.42.30.png](assets/Screenshot_2023-12-10_at_22.42.30.png)

# **Сравнение метрик**

По итогам WMT Metrics Shared Task 2022 [опубликован рейтинг](https://www.statmt.org/wmt22/pdf/2022.wmt-1.2.pdf) ряда традиционных, нейросетевых, а также безреференсных метрик (их названия помечены звезочкой – *) по степени их корреляции с человеческой оценкой качества перевода:

Традиционные метрики (BLEU, chrF) существенно отстают от наиболее продвинутых нейросетевых. Лучше всего показали себя референсные нейросетевые метрики. Топовая метрика – MetricX XXL, а из задокументированных метрик наиболее адекватна метрика COMET в версии 2022 года ([Rei et al., 2022](https://aclanthology.org/2022.wmt-1.52.pdf)).

Безреференсные метрики хуже топовых референсных нейросетевых, но большинство из них опережает традиционные.

![Сравнение метрик оценки качества по степени корреляции с человеческой оценкой качества перевода](assets/Screenshot_2023-12-10_at_22.43.45.png)

Сравнение метрик оценки качества по степени корреляции с человеческой оценкой качества перевода

# Бенчмарки на примере SuperGLUE

В последние годы развитие языковых моделей привело к эволюции метрик оценки качества. Тест GLUE, впервые представленный в начале 2019 года, призван предложить одночисловую метрику, позволяющую количественно оценить производительность языковой модели при выполнении различных типов задач понимания. Но по мере того, как с 2020 года исследования над большими языковыми моделями резко возросли, тест GLUE стал довольно устаревшим, поскольку модели превосходили результаты неспециалистов в этом тесте, требуя более сложной метрики.

Исследователи Facebook совместно с Google DeepMind, Вашингтонским университетом и Нью-Йоркским университетом в конце 2019 года представили SuperGLUE — серию тестовых задач для измерения производительности искусственного интеллекта, распознающего речь.

SuperGLUE была создана на основе нейронной сети Google BERT. Производительность BERT, как сообщает VentureBeat, превзошла такие модели, как MT-DNN от Microsoft, XLNet от Google и RoBERTa от Facebook, которые обеспечивают высокую производительность — выше среднего базового уровня человека.

Предшественником SuperGLUE стал бенчмарк General Language Understanding Evaluation (GLUE), который был разработан в апреле 2018 года исследователями из Нью-Йоркского университета, Университета Вашингтона и компанией DeepMind. SuperGLUE на порядок сложнее GLUE и будет, по планам разработчиков, стимулировать создание моделей, способных воспринимать более тонкие речевые нюансы.

SuperGLUE будет включать в себя восемь задач для проверки способности системы следовать за мыслью, распознавать причину и следствие и отвечать на вопросы «да» или «нет».

> «Современные системы ответов на вопросы ориентированы на пустячные вопросы, например, есть ли у медузы мозг. Эта система идет дальше, требуя от машин проработать подробные ответы на открытые вопросы, такие как «Как медузы функционируют без мозга?» — говорится в сообщении Facebook.
> 

Чтобы помочь исследователям создать надежный ИИ для понимания языка, Нью-Йоркский университет также выпустил обновленную версию Jiant — универсального инструмента для понимания текста. Jiant настроен для работы с HuggingFace PyTorch BERT и OpenAI GPT, а также тестами GLUE и SuperGLUE.

Подзадачи, включенные в SuperGLUE, следующие:

- Логические вопросы: BoolQ — это задача, требующая ответов на вопросы. Он состоит из короткого отрывка из статьи в Википедии и вопроса «да/нет» о модели. Выполнение оценивается с точностью.
- CommitmentBank: CB состоит из коротких текстов, содержащих хотя бы одно встроенное предложение. Задача включает в себя определение уровня приверженности автора истинности предложения. Данные получены из различных источников, таких как Wall Street Journal, Британский национальный корпус и Switchboard. Учитывая дисбаланс данных, для оценки используются точность и невзвешенный средний балл F1.
- Выбор вероятных альтернатив: COPA — это задача причинного рассуждения. Системе представлено предложение, и она должна распознать причину или следствие из двух вариантов. Примеры тщательно отобраны из сообщений в блогах и фотоэнциклопедии. Метрикой оценки является точность.
- Понимание чтения нескольких предложений: MultiRC — это задача контроля качества, включающая контекстный абзац, связанный вопрос и несколько потенциальных ответов. Система должна классифицировать ответы как истинные или ложные. Метрики оценки включают оценку F1 по всем вариантам ответа и точное совпадение набора ответов на каждый вопрос.
- Понимание прочитанного с набором данных для рассуждений на основе здравого смысла: ReCoRD — это задача контроля качества с множественным выбором, включающая новостную статью и вопрос с замаскированным объектом. Система должна предсказать замаскированный объект на основе предоставленных вариантов. Оценка включает в себя расчет максимального балла F1 на уровне токена и его точное соответствие.
- Распознавание текстового следствия: наборы данных RTE взяты из ежегодных соревнований по текстовому следствию. Данные нескольких итераций конкурса объединяются в задачу классификации двух классов: entailment и not_entailment. Метрикой оценки является точность.
- Слово в контексте: WiC — это задача бинарной классификации, включающая пары предложений и многозначное слово. Задача – определить, имеет ли слово одинаковый смысл в обоих предложениях. Используемый показатель оценки — точность.
- Задача по схеме Винограда: WSC — это задача по разрешению кореференции, которая требует определения правильного референта местоимения из списка именных фраз в предложении. Задача требует использования здравого смысла и оценивается с использованием точности.

# TODO:

> [https://metaverse-imagen.gitbook.io/ai-tools-research/ai-resources/llm-benchmarks-and-tasks/mmlu-massive-multitask-language-understanding](https://metaverse-imagen.gitbook.io/ai-tools-research/ai-resources/llm-benchmarks-and-tasks/mmlu-massive-multitask-language-understanding)
> 

# Оценки с привлечением людей

- Ранжирование списка моделей по качеству на конкретной задаче / данных
- Pairwise comparison
- Метод Брэдли-Терри

### LLM как эксперт

Огромный потенциал LLM-моделей позволяет им не только решать лучше людей некоторые задачи, но и выступать в роли эксперта для разметки данных.

GPT-3 может использоваться вместо краудсорсинга для задачи разметки текстов [(Ding, 2022)](https://arxiv.org/pdf/2212.10450.pdf). Авторы приходят к выводу, что при ограниченном бюджете на разметку, качество обученных моделей будет не хуже, чем при разметке реальными людьми. 

Также есть исследования ****[(Alizadeh, 2023)](https://arxiv.org/pdf/2307.02179.pdf), что open-source LLM модели показывают сравнимые с ChatGPT результаты, при этом при наличии вычислительных мощностей, могут быть значительно более эффективными инструментами разметки в аспектах стоимости и скорости.

Кроме того, человеческая разметка на деле может оказаться разметкой с помощью LLM моделей, так как краудсорсеры могут использовать LLM-модели для повышения своей производительности. При задаче разметки текста в Amazon MTurk до 33-46% ответов людей могут быть на самом деле ответами LLM-моделей ****[(Veselovsky, 2023)](https://arxiv.org/pdf/2306.07899.pdf). Это может служить мотивацией в большей степени исследовать методы оценки с помощью LLM, так как краудсорсинг может перестать быть разметкой с помощью людей на самом деле.

Например, в бенчмарке [rulm-sbs2](https://github.com/kuk/rulm-sbs2) в качестве эксперта используется GPT-4. И приводится список причин, по которым была выбрана LLM вместо человеческой разметки.

- При ручной проверке 100 случайных пар ответов из оценок толокеров из проекта [Saiga](https://huggingface.co/datasets/IlyaGusev/rulm_human_preferences) оказывается, что в 70 случаях ответы толокеров и GPT-4 совпали, 15 раз скорее были правы краудсорсеры, 15 раз скорее была права GPT-4
- При этом у GPT-4 лучше получается отвечать на вопросы, требующие экспертизы, например, решение задач по программированию или математике

Таблица со сравнением GPT-4 и краудсорсеров из проекта [Toloka](https://toloka.ai) из описания бенчмарка:

|  | Crowd | GPT-4 |
| --- | --- | --- |
| Цена | 67$ за 1000 заданий с перекрытием 5 | 45$ за 1000 заданий |
| Скорость | 1 час модерация проекта + ~250 заданий в час | ~3600 заданий в час |
| Качество | Краудсорсеры ошибаются в ~15% заданий.
Сложно оценить задания про программирование, математику | GPT-4 ошибается в ~15% заданий.
Не может оценить задания, где у самой GPT4 низкое качество: задания на знание русского языка, задания на запретные темы |
| Предвзятость | Краудсорсеры беспристрастны | GPT-4 скорее нравятся ответы GPT-4 и Turbo |
| Интерпретация | Краудсорсеры не объясняют решение | Может объяснить решение |

[Оценка качества ccdabdf3085f42fc8e789db894b7cfc7.md](%D0%9E%D1%86%D0%B5%D0%BD%D0%BA%D0%B0%20%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0%20ccdabdf3085f42fc8e789db894b7cfc7/%25D0%259E%25D1%2586%25D0%25B5%25D0%25BD%25D0%25BA%25D0%25B0_%25D0%25BA%25D0%25B0%25D1%2587%25D0%25B5%25D1%2581%25D1%2582%25D0%25B2%25D0%25B0_ccdabdf3085f42fc8e789db894b7cfc7.md)

Можно сделать вывод, что при сравнимом качестве оценка с помощью GPT-4 дешевле и быстрее. Также имеет смысл совмещать оценку с помощью людей и LLM, так как природа ошибок разная, а также необходимо устранять предвзятость LLM в оценке. Также LLM можно использовать для объяснения разницы в примерах, вызывающих неопределенность у толокеров. 

Потенциально могут появляться специализированные LLM или фреймворки для разметки данных и оценки качества. Например, AnnoLLM [(He, 2023)](https://arxiv.org/pdf/2303.16854.pdf) предлагает подход по улучшению процесса разметки данных с помощью ChatGPT.

### Проблемы использования LLM вместо краудсорсинга

- Автоматическая оценка с помощью LLM моделей поддается атакам, которые искажают результаты сравнения [(Wang, 2023)](https://arxiv.org/pdf/2305.17926v1.pdf). Исследованная атака помогала достичь систематической предвзятости за счет перемешивания примеров для оценки внутри контекста LLM модели
- LLM модели отдают большее предпочтение стилю ответов, чем верности фактов [(Gudibande, 2023)](https://arxiv.org/pdf/2305.15717.pdf)
- Текущие фреймворки и датасеты могут не отражать картины, наблюдаемой при реальном использовании модели, поэтому не должны применяться для решения о выводе новых моделей в production использование
- LLM отдают большее предпочтение тем моделям, с которым они делят тренировочный датасет
- Оценка с помощью LLM не проверяет такие особенности, как грубость и токсичность, так что не может служить основанием для внедрения модели в production

### Бенчмарки для оценки качества

- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) — фреймворк для автоматической оценки моделей с помощью LLM
    - Большой выбор моделей-оценщиков
    - Leaderboard из 12+ LLM
- [rulm-sbs2](https://github.com/kuk/rulm-sbs2) — бенчмарк для ранжирования LLM по качеству по качеству работы на русском языке
    - GPT-4 как эксперт для оценки
    - 500 заданий из 15+ категорий
    - Задания — переведенные на русский Alpaca + Vicuna + часть Arena