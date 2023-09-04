from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

raw_datasets = load_dataset("squad")
model_checkpoint = "bert-base-cased"  # The model is based on BERT
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


class ParaSet:
    def __init__(self, max_length, stride):
        self.max_length = max_length
        self.stride = stride

    def __iter__(self):
        return iter((self.max_length, self.stride))


def model_checkpoint(mod_checkpoint):
    tokenizers = AutoTokenizer.from_pretrained(mod_checkpoint)
    return tokenizers


def preprocess_validation(dataset):
    questions = [q.strip() for q in dataset["question"]]
    context = [q.strip() for q in dataset["context"]]
    max_length, stride = ParaSet(384, 128)
    inputs = tokenizer(
        questions,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])): # number of the features
        sample_idx = sample_map[i]
        example_ids.append(dataset["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
    inputs["example_id"] = example_ids
    return inputs



