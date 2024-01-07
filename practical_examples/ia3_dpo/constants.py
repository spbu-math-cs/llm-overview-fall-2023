DATASET_NAME = "MoritzLaurer/multilingual-NLI-26lang-2mil7"
RANDOM_STATE = 42
SFT_TRAIN_SIZE_RATIO = 0.5
TOKENIZED_TEXT_MAX_LEN = 128
SUBSTITUTION_MAP = {
    # eng --> ru
    "c": "с",
    "o": "о",
    "y": "у",
    "C": "С",
    "O": "О",
    "Y": "У",
    # ru --> eng
    "с": "c",
    "о": "o",
    "у": "y",
    "С": "C",
    "О": "O",
    "У": "Y",
}
EPOCH_NUM = 5
SAVE_STEPS = 50
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
SFT_LEARNING_RATE = 1e-4
INSTRUCTION_PROMPT = 'Replace the English letter c with the Russian letter с, the English letter o with the Russian letter о, the English letter y with the Russian letter у, and vice versa in the following text.\n\nText: {}\n\nAnswer: '
BATCH_SIZE = 16
DPO_BETA = 0.1
DPO_LEARNING_RATE = 1e-4
