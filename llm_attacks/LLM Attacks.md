# Атаки на LLM

# Token Manipulation

Пусть нам на вход подается какая-то последовательность токенов. Мы можем изменить небольшую часть входных токенов таким образом, чтобы заставить модель выдавать неверные ответы, но при этом желательно, чтобы семантическое значение входа не изменялось. Например, можно заменить некоторые из входных токенов синонимами. Такие атаки называются "Token Manipulation", и они работают в концепте BlackBox. Весь Token Manipulation можно разделить на 4 этапа, как это сделано например в статье "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP" ([arxiv](https://arxiv.org/abs/2005.05909)).

Для начала определим критерий успешного взлома (Goal function). По выходам атакованной модели определяет успешно ли была произведена атака. Примеры таких критериев:

- BLEU score ниже, чем у исходной модели;
- ошибочная классификация (ожидался один класс, а был получен другой);
- несовпадение текстов (ни одно слово не совпадает по каждой позиции между ожидаемым и полученным выходами).

Также, нам нужен набор ограничений, который определяет, является ли примененная ко входному токену модификация допустимой. Примеры таких ограничений:

- минимальное косинусное расстояние между эмбеддингами токенов;
- минимальное косинусное расстояние между эмбеддингами предложений;
- совпадение части речи у исходного и атакующего токенов.

Третьим этап можно определить методы по модификации токенов. Получая на вход токен из исходного предложения, или все предложение генерирует атакующий токен / предложение. Примеры:

- замена на токен с ближайшим по косинусной мере эмбеддингом;
- замена на "омонимичный" токен (заменить букву `l` на `1`);
- случайная перестановка токенов в предложении;
- правила ЕСЛИ-ТО (например, статья SEAR: Semantically Equivalent Adversarial Rules for Debugging NLP Models ([ссылка](https://sameersingh.org/files/papers/sears-acl18.pdf))).

Последним этапом будет фильтрация атакующих кандидатов. Как выбирать наиболее перспективные атакующие последовательности. Ведь даже если самая перспективная последовательность не прошла проверку на успешную атаку, то ее можно взять за новую последовательность и зациклившись пройти еще один шаг атаки. Примеры:

- Генетический алгоритм;
- Beam Search;
- Полный перебор.

Авторы статьи приводят еще и такой красивый интерфейс для визуализации результатов.

![Screen Shot 2023-12-11 at 22.00.39.png](assets/Screen_Shot_2023-12-11_at_22.00.39.png)

---

В 2020 году вышли еще две статьи: “Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment” ([arxiv](https://arxiv.org/abs/1907.11932)) и “BERT-ATTACK: Adversarial Attack Against BERT Using BERT” ([arxiv](https://arxiv.org/abs/2004.09984v1)), по сути об одном и том же: как можно атаковать языковые модели используя эмбеддинги и Semantic Similarity между предложениями. В обеих статьях можно заметить все те же 4 этапа, которые были описаны ранее.

Сетап у нас следующий: есть BlackBox языковая модель, на вход которой мы подаем текст, а получаем $f(x)$ - $N$-мерный вектор вероятностей каждого класса (суммируется в 1). Наша задача - найти такую модификацию входного текста, для которой $argmax(f(x_{init})) ≠ argmax(f(x_{modified}))$.

Интуитивно, наибольший интерес представляет модификация тех токенов, к которым модель наиболее чувствительна. Осталось как-то определить такие токены, для этого определим “важность” слова $w_i$ из входного текста как указано в формуле ниже.

![Screen Shot 2023-12-11 at 21.03.57.png](assets/Screen_Shot_2023-12-11_at_21.03.57.png)

Через $f(x_{\setminus w_i})$ мы обозначаем последовательность токенов $x$ без токена $w_i$. Тогда, если предсказанный класс у исходного предложения $y = argmax(f(x))$  и предсказанный класс у модифицированного предложения $\overline{y} = argmax(f(x_{\setminus w_i}))$  совпадают, то мы возьмем разницу между этими сигмоидами как значение “важности”. Иначе, в качестве важности возьмем сумму разниц сигмоид как на классе, предсказанном по исходной последовательности, так и на классе, предсказанном по модифицированной последовательности.

Далее мы отсортируем токены от наиболее к наименее важному. Для каждого токена $t$ из отсортированных мы извлечем $M$ токенов-синонимов из нашего словаря, выбрав такие токены, для эмбеддингов которых косинусное расстояние с эмбеддингом токена $t$ будет наименьшей. 

Попробуем подставить на место токена $t$ каждый из его токенов-синонимов. Если после подстановки семантическое расстояние модифицированной последовательности будет отличаться от исходной последовательности больше чем на $\varepsilon$, то такого кандидата мы выкидываем, а иначе посмотрим какое предсказание мы получим от атакуемой модели, используя модифицированную последовательность. Если модель ошиблась в предсказании, то можно считать, что мы решили задачу и выдать модифицированную последовательность. Если же ни один токен не справился с атакой, то заменим его тем токеном-финалистом, на котором модель была ближе всего к ошибке, т.е. для которого значение сигмоиды правильного класса было бы минимальным.

![Screen Shot 2023-12-11 at 21.37.03.png](assets/Screen_Shot_2023-12-11_at_21.37.03.png)

Такой подход к атаке на языковые модели (в том числе BERT) позволяет снизить качество классификации с более чем 80% точности до 10-20% (как паказано в таблице выше). Это очень серьезная разница в качестве, ведь модель с качеством 10-20% на задаче бинарной классификации можно назвать неприменимой. Однако, можно заметить узкое место данного алгоритма - подсчет значимости токенов. Как только мы оградим атакующего от значений выходов сигмоиды, то производить такие атаки станет заметно сложнее.

# Атаки с использованием градиентов

В white-box случае у нас есть полный доступ к параметрам и архитектуре модели. Поэтому мы можем использовать различные методы оптимизации (в своём большинстве градиентный спуск) для обучения наиболее эффективным атакам. Атаки, основанные на градиенте, работают только в случае white-box (однако, есть результаты, удачно применяющие результаты этих атак на black-box моделях), например, для открытых исходных языковых моделей (LLMs).

**Исследование одного из подходов в этом направлении атак было проведено в [следующей статье:](https://arxiv.org/abs/2307.15043)** 

Исследовались универсальные провоцирующие суффиксы, добавляемые к входному запросу. Особое внимание уделялось запросам, на которые модель должна отказываться отвечать, в данном случае - запросам с недопустимым содержанием, таким как советы по совершению преступлений. Суть этих суффиксов состояла в том, чтобы привести языковые модели к выдаче утвердительных ответов на запросы, на которые следовало бы отказать. Другими словами, в ответ на злонамеренный запрос модель должна выдать ответ, начинающийся, например, на  "Конечно, вот как..." (или другие **утвердительные** **стандартные начала** предложений)

На месте !!! стоит оптимизируемый суффикс

![zou (1).png](assets/zou_(1).png)

Для оптимизации использовался метод жадного координатного градиентного (GCG) поиска, направленный на жадное нахождение кандидата, который максимально снизит потери среди всех возможных однотокенных замен. Была применена стратегия поиска токенов на основе градиента, аналогичная UAT и AutoPrompt, для нахождения лучших кандидатов на каждый токен, каждый из которых ассоциирован с наибольшим отрицательным градиентом потерь. 

![Снимок экрана 2023-12-11 в 14.21.23.png](assets/1.png)

Для каждого токена в суффиксе мы находим топовые значения с наибольшим отрицательным градиентом относительно потерь NLL (negative log-likelihood) языковой модели. При этом индексация начинается с 1. Затем из множества опций случайным образом выбираются кандидаты на замену токена, и из них выбирается тот, у которого наибольший  NLL, чтобы установить его в качестве следующей версии токена. Этот процесс в основном заключается в том, чтобы. Эта процедура продолжается до успешного срабаатывания суффикса на текущем примере. Было выяснено, что такой пошаговый подход работает лучше, чем попытка оптимизировать весь набор запросов сразу.

![Снимок экрана 2023-12-11 в 14.21.03.png](assets/2.png)

### Сравнение с другими подходами подбора :

![Снимок экрана 2023-12-11 в 13.43.42 (1).png](assets/4.png)

### Результаты:

Удивительным является то, что полученные суффиксы пригодны для проведения атак на проприетарные модели

![Снимок экрана 2023-12-11 в 13.58.42.png](assets/3.png)

## Jailbreak Prompting

Black-Box атака с помощью вредной последовательности сообщений от пользователя.

По версии авторов статьи "Jailbroken: How Does LLM Safety Training Fail?” (Wei et al., 2023) есть две причины возникновения уязвимости:

- Конкурирующие цели — цель модели отличается от цели системы безопасности
- Ошибки обобщения — в обучающем множестве модели встречаются out-of-domain примеры для системы безопасности

Рассмотрим каждую причину подробнее.

### Конкурирующие цели

При обучении большой языковой модели, её могут учить выполнять инструкции пользователя, это будет первоочередная цель. После получения достаточного для внедрения качества — модель начинают готовить к эксплуатации в реальном мире. В этот момент модель получает ряд ограничений, например, модели запрещается затрагивать острые политические темы, выдавать по запросу рецепты запрещенных веществ и так далее.

В этот момент, если пользователь произведет запрос, вынуждающий модель ответить опасным с точки зрения системы безопасности — образуется конкуренция между первоначальной целью модели исполнить инструкцию и целью системы безопасности не допустить небезопасных ответов.

Конкретные примеры зависят от реализации системы безопасности:

- Например, если система безопасности каким-то образом оценивает опасность префикса токенов ответа — можно дать модели инструкцию внедрить определенный безопасный префикс — например, начать с “Безусловно, вот”. Такой трюк работал в первых публичных версиях GPT-4:
    
    ![Снимок экрана 2023-11-23 в 18.18.53.png](assets/6.png)
    
- Если система безопасности реализована через дообучение модели на опасных запросах, возможно подавить дообучение. Например, можно убедить модель, что ей ни в коем случае нельзя ответить отказом. Или если при дообучении целевые ответы были написаны в официальном стиле, можно попросить модель ответить неформально или используя только короткие слова. Основная цель таких способов — заставить модель выдать ответ из другого пространства, нежели пространство ответов в дообучении.

### Ошибки обобщения

Широко изучается поведение нейронных сетей при обработке out-of-domain данных — данных не из тренировочного распределения. Чаще всего, качество работы модели на таких данных серьезно падает, относительно целевого распределения. 

Если после обучения системы безопасности — не так важно что именно она из себя представляет: дообучение, соседнюю модель или просто набор регулярных выражения — окажется, что в тренировочном множестве базовой модели были out-of-domain примеры относительно обученного “опасного” распределения, мы не можем ничего гарантировать про точность системы безопасности.

Широко исследуются методы определения out-of-domain семплов для моделей, в зависимости от типа моделей можно выбрать один из методов и просканировать обучающее множество на предмет таких семплов. Но к сожалению, условия для обучения распознавания out-of-domain семплов практически не выполнимы в реальной жизни, как показали в статье “Is Out-of-Distribution Detection Learnable?” (Fang et al., 2022). Так что в нашей системе безопасности непременно останутся возможности для атаки с помощью ошибки генерализации системы безопасности.

Например, opensource модель Claude обучалась в том числе на закодированных в base64 текстах и обучилась правилам кодирования. При этом не учесть вектор атаки через base64 кодирование несложно. Как и не участь любой другой возможный шифр или формат выхода, который модель может изучить из тренировочного датасета — шифр Цезаря, ответ в формате JSON, YAML, и т.п.

![Снимок экрана 2023-11-23 в 18.36.26.png](assets/5.png)

### Наблюдения

В статье (Greshake et al., 2023) приводятся также следующие наблюдения о возможных векторах атак:

- Если внедрить в инструкции модели описание цели атаки, модель может автономно её совершать, используя известные способы. Это может сделать модель крайне опасным инструментов в руках мошенников, так как будет работать относительно автономно и хорошо масштабироваться.
- Внедрение контекста в инструкции модели может привести к проекции контекста на ответы. Например, если внедрить в инструкции модели политический контекст одной из партий без прямых инструкций, модель может проецировать какие-то идеи из контекста на свои ответы.
- В инструкциях модели можно просить не воспроизводить информацию из достоверных источников и наоборот просить воспроизводить информацию только из выделенной группы источников. В таком случае модель может начать выдавать дизинформацию или принять какую-то определенную точку зрения, при этом достаточно лишь перечислить список источников.

# ****Humans in the Loop****

Одной из потенциальных идей для подхода к генерации датасетов с атакой на языковые модели — добавить человеческий фактор. В таком случае важным является то, как человек будет это делать. Один из таких подходов был предложен [Wallace et al. (2019)](https://www.notion.so/todo). В рамках данной исследовательской работы был предложен интерфейс для помощи человеку атаковать языковые модели, а также в результате был сгенерирован датасет, который продемонстрировал интересные результаты.

Задачей было отвечать на вопросы в формате "Своей игры". Вопросы построены таким образом, что более эрудированный человек должен успеть догадаться раньше остальных: вопрос уточняется постепенно и чем дольше он звучит -- тем более конкретным он становится. Задачей человека в данной работе было либо заставить модель ответить на вопрос неправильно, либо отсрочить ответ как можно сильнее. В генерации участвовало два типа моделей: RNN, а также модель [IR](https://link.springer.com/chapter/10.1007/978-3-319-94042-7_9), основанная на Elastic Search.

![image.png](assets/image.png)

Интерфейс позволял человеку видеть уверенность модели в других ответах, а также подсвечивал важность для модели слов в вопросе. Это было апроксимированно с помощью градиента по отношению к эмбеддингу токена.

Для измерения результатов была взята также модель [DAN](https://aclanthology.org/P15-1162/) (Deep Averaging Network), чтобы показать, как датасет сгенерированный для конкретных моделей может распространяться на другие.

![image-1.png](assets/image-1.png)

Для измерения результатов было сделано два раунда: только с IR-based датасетом, а потом еще и с RNN-based. Можно заметить, что действительно датасеты неплохо наспространились на DAN. При этом видно, что IR-based датасет хорошо взламывает другие модели, при этом сама модель робастна к датасету для RNN.

![image-2.png](assets/image-2.png)

Также были проведены исследования данных датасетов на людях (так как их генерировали люди, они человекочитаемы). Можно заметить, что человек лучше справляется с IR-based датасетом, чем с обычным, сгенерированным под человека. Это можно объяснить тем, что датасет, нацеленный на обман модели не обязательно будет полностью удовлетворять тому условию, с которым обычно состовляются вопросы для “Своей игры”, что потенциально может дать человеку преимущество.

![image-3.png](assets/image-3.png)

Также было проведено исследование на [state-of-the-art системы](https://arxiv.org/abs/1803.08652) для данной задачи. Несмотря на то что генерация датасетов не была на нее нацелена, на них все равно видна заметная деградация.

Похожий подход был разработан [Ziegler et al. (2022)](https://arxiv.org/abs/2205.01663). В данной работе рассматривалась задача классификации небезопасного контента, человек в данном случае так же, через интерфейс, пытался обмануть классификатор. Помимо подсвечивания токенов аналогичным прошлой статье способом, также была добавлена возможность предложения новых токенов, которые генерировались с использованием модели BERT. Данный подход позволил ускорить работу человека.

![Untitled](assets/Untitled.png)

Также похожими подходами были сгенерированы и другие датасеты:

- [BAD Dataset](https://aclanthology.org/2021.naacl-main.235/) -- около 2500 опасных диалогов с языковой моделью
- [Red-teaming dataset, Anthropic](https://arxiv.org/abs/2209.07858) -- приблизительно 40 тысяч атак, полученных из общений с LLM. В качестве одного из результатов авторы выяснили, что RLHF модели тяжелее взламывать, чем остальные.

# LLM Robustness

Одним из простых и интуитивно понятных способов защиты модели от состязательных атак является явное указание модели нести ответственность и не создавать вредоносный контент. Это может значительно снизить вероятность успеха Jailbreak атак, но имеет побочные эффекты для общего качества модели из-за того, что модель действует более консервативно (например, для творческого письма) или неправильно интерпретирует инструкции в некоторых сценариях (например, классификация безопасных и небезопасных).

Рассмотрим способ защиты, предложенный [предложенный исследователями из Майкрософт](https://assets.researchsquare.com/files/rs-2873090/v1_covered_3dc9af48-92ba-491e-924d-b13ba9b7216f.pdf?c=1686882819) для защиты СhatGPT от Джейлбрейк-атаки.

![Screenshot 2023-12-10 at 22.57.19.png](assets/Screenshot_2023-12-10_at_22.57.19.png)

На изображении можем увидеть пример Джейлбрек-атаки и предлагаемую технику защиты:

- В части **a** представлен аморальный запрос пользователя без джейлбрейк атаки. Мы можем видеть, что модель справляется с такими запросами и способна предотвратить создание вредоносных ответов;
- В части **b** Джейлбрейк атака может обойти моральные ограничения модели используя специальный Джейлбрейк-запрос, чтобы обойти запреты ChatGPT и генерировать ответ на вредоносные запросы;
- В части **c** демонстрируется использование подхода, который описывается в статье: **System-Mode Self-Reminder,** который на системном уровне оборачивает пользовательский запрос и напоминает модели о том, что ей следует быть отвественное при генерации ответа.

## A**dversarial training**

Самый распространенный способ снизить риски состязательных атак — это обучение модели на этих образцах атак, известное как состязательное обучение. Это считается самой сильной защитой, но приводит к компромиссу между надежностью и производительностью модели. 

В эксперименте [Jain et al. В 2023 году](https://arxiv.org/pdf/2309.00614v2.pdf) они протестировали две схемы состязательного обучения: 

1. Запускать градиентный спуск по вредоносным подсказкам в сочетании с ответом «Мне очень жаль. В качестве…»; 
2. Выполнить один шаг спуска при отказе от ответа и шаг подъема при плохом ответе за каждый тренировочный шаг.

В результате экспериментов было выяснено, что второй метод оказывается совершенно бесполезным, поскольку качество генерации модели сильно ухудшается, а падение успешности атаки незначительное.