import tensorflow as tf
import collections
import numpy as np
import torch
from transformers import TFAutoModelForQuestionAnswering, AutoModelForQuestionAnswering
from training import preprocess_training
from datasets import load_dataset
from transformers import AutoTokenizer
from verification import preprocess_validation
import evaluate
from tqdm.auto import tqdm
from transformers import pipeline
from huggingface_hub import notebook_login
from transformers import DefaultDataCollator
from huggingface_hub import notebook_login
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
from transformers.keras_callbacks import PushToHubCallback
from huggingface_hub import login
#
raw_datasets = load_dataset("squad")
#
train_dataset = raw_datasets["train"].map( # Batch referencing functions to process datasets
    preprocess_training,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

validation_dataset = raw_datasets["validation"].map(    # Batch referencing functions to handle validation sets
    preprocess_validation,
    batched=True,  # batch file
    remove_columns=raw_datasets["validation"].column_names,
)
# #
# #
# small_eval_set = raw_datasets["validation"].select(range(100))
# trained_checkpoint = "distilbert-base-cased-distilled-squad"
#
# tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
# eval_set = small_eval_set.map(
#     preprocess_validation,
#     batched=True,
#     remove_columns=raw_datasets["validation"].column_names,
# )
# #
# #
model_checkpoint = model_checkpoint = "bert-base-cased"  # The model is based on BE
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# #
# #
# eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping", "token_type_ids"])
# eval_set_for_model.set_format("torch")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # GPU training
# batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
# trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
#     device
# )
# # with torch.no_grad():
# #     outputs = trained_model(**batch)
# #
# # start_logits = outputs.start_logits.cpu().numpy()
# # end_logits = outputs.end_logits.cpu().numpy()
# #
# # example_to_features = collections.defaultdict(list)
# # for idx, feature in enumerate(eval_set):
# #     example_to_features[feature["example_id"]].append(idx)
# #
# # n_best = 20
# # max_answer_length = 30
# # predicted_answers = []
# #
# # for example in small_eval_set:
# #     example_id = example["id"]
# #     context = example["context"]
# #     answers = []
# #
# #     for feature_index in example_to_features[example_id]:
# #         start_logit = start_logits[feature_index]
# #         end_logit = end_logits[feature_index]
# #         offsets = eval_set["offset_mapping"][feature_index]
# #
# #         start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
# #         end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
# #         for start_index in start_indexes:
# #             for end_index in end_indexes:
# #                 # Skip answers that are not fully in the context
# #                 if offsets[start_index] is None or offsets[end_index] is None:
# #                     continue
# #                 # Skip answers with a length that is either < 0 or > max_answer_length.
# #                 if (
# #                     end_index < start_index
# #                     or end_index - start_index + 1 > max_answer_length
# #                 ):
# #                     continue
# #
# #                 answers.append(
# #                     {
# #                         "text": context[offsets[start_index][0] : offsets[end_index][1]],
# #                         "logit_score": start_logit[start_index] + end_logit[end_index],
# #                     }
# #                 )
# #
# #     best_answer = max(answers, key=lambda x: x["logit_score"])
# #     predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
# #
# # metric = evaluate.load("squad")
# #
# # theoretical_answers = [
# #     {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
# # ]
# #
# # print(predicted_answers[0])
# # print(theoretical_answers[0])
# #
# # mark = metric.compute(predictions=predicted_answers, references=theoretical_answers)
# # print(mark)


from transformers import TrainingArguments
from transformers import Trainer
from tqdm.auto import tqdm


model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
mark = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])