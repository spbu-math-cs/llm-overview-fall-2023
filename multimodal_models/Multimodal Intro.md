## Intro to Multimodal Models
_"... a large gap persists between the capabilities of LLMs and true humanlike intelligence. 
This is partially because humans perceive a variety of sensory inputs while LLMs are typically
restricted to Language..."_ - Microsoft Azure Cognitive Services Research (MACSR)

"_The convergence of text, visual, and audio data is a key step towards human-like
artificial intelligence_" - MACSR.
### Selling Multimodal

#### Ilya Sutskever, OpenAI chief scientist
В [недавнем интервью](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52092/?ncid=so-yout-561702)
Jensen Huang Илья рассказал про свое видение Multimodal направления развития ML. 
Две причины почему расширение LLM на другие модальности (в основном vision) кажется Илье интересным:

 - Humble: это просто полезно.
   Мир, как его воспринимает человек, очень визуален. 
   Поэтому полезность нейросеток без визуальной модальности не такая большая, 
   как могла бы быть если бы сетки видели то, что видят люди. 
   Кстати, в GPT-4 добавили vision, теперь это мультимодальная модель.
 - Not as clear-cut as it may seem: дополнительно учась на изображениях, 
   модель узнает и поймет больше, чем пользуясь одним текстом.
   Люди за всю жизнь слышат не более миллиарда слов, 
   и для людей важно получать информацию из всех доступных модальностей, 
   причем он считает, что мы выучиваем гораздо больше из визуальной модальности. 
   То же верно для сеток, с тем исключением, что сетки успевают увидеть триллионы слов, 
   поэтому им проще узнать много о мире из одного лишь текста. 
   Тем не менее, это неэффективно, 
   и можно гораздо быстрее понять мир пользуясь изображениями и видео.
   То есть важное преимущество, которое приносит Multimodal - 
   благодаря новым каналам информации
   становится проще обучиться понимать мир. А также приблизить понимание
   к человеческому.

Он приводит пример с GPT-4, которая начала решать математический тест 
c диаграммами в условиях сильно лучше после добавление Vision.
Илья говорит, что кроме понимания мира есть и другие аспекты:
иметь возможность рассуждать ("to reason") визуально и коммуницировать визуально -
это очень мощные вещи.
   
#### New capabilities
С Multimodal LLM-агенты становятся приспособлены к задачам, 
которые не удавалось качественно решить в Unimodal сеттинге. 
Пример: Document Editing.

<img src="assets/doc_editing.png" width="600">

[SOTA-решение](https://arxiv.org/pdf/2212.02623.pdf) (на 13 марта) 
обрабатывает документ по трем модальностям - visual, text и layout 
(разметка, тоже своего рода модальность). На вход подается изображение,
из него экстрагируется layout и text, после чего трансформер обучается на трех
связанных модальностях. Ожидаемо
подход имеет преимущество перед наивным ViT с masked unit modelling.

### Состояние области в соответствии с недавним [обзором](https://arxiv.org/pdf/2309.10020.pdf)
Иллюстрация текущего положения дел из [обзорной статьи](https://arxiv.org/pdf/2309.10020.pdf) 
про Multimodal c упором на vision - параллель между прогрессом Textual и Multimodal моделей.

<img src='assets/models_compared.png' width='600'>

Главные компоненты задачи, выделенные в той статье (опять же с фокусом на vision)

<img src='assets/questions.png' width='600'>

И классификация vision-ориентированных моделей по функции

<img src='assets/paper_structure.png' width='600'>


### Central challenges

 - Data Collection and Labeling.
For instance, gathering a balanced set of user posts that contain varied data types – 
text, images, videos – is no small feat. Accurately labeling this data and making sure 
modalities are aligned is also typically harder than Unimodal data processing.
 - Model Complexity.
Multimodal learning models are inherently complex due to their need to process 
and integrate multiple data types.
 - Data Fusion. 
How and when to combine different modalities into one flow? Early fusion might cause loss of unique characteristics of individual modalities, 
while late fusion might not capture the correlations between modalities adequately.
Moreover, depending on the task you've got to find the right balance between the importance of 
modalities - e.g.
textual data (genres, tags), visual data (posters), and audio data (soundtracks, dialogue) 
for the recommendation task.

