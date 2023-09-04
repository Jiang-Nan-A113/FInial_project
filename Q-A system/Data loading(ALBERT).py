import transformers
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AlbertModel

raw_datasets = load_dataset("squad")
# train = raw_datasets["train"][0]
# model_checkpoint = "albert-base-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#
# question = raw_datasets["train"][2: 6]["question"]
# question = [q.strip() for q in question]
# print(question)
#
# context = raw_datasets["train"][2: 6]["context"]
#
# inputs = tokenizer(
#     question,
#     context,
#     max_length=100,
#     truncation="only_second",
#     stride=50,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
# )
#
# answers = raw_datasets["train"][2: 6]["answers"]
# print(answers)
# start_positions = []
# end_positions = []
#
#
# for i, offset in enumerate(inputs["offset_mapping"]):  # 'i' is the number of in the inputs, offset is the embedding
#     # number of the sentences
#     # print(offset)
#     inputs_idx = inputs["overflow_to_sample_mapping"][i]  # This is to clear the features is from which sentences
#     answer = answers[inputs_idx]  # so we can find the answer of the feature
#     start_char = answer['answer_start'][0]
#     # print(start_char)
#     end_char = answer['answer_start'][0] + len(answer["text"][0])  # start position + len of the sentence = end position
#     sequence_ids = inputs.sequence_ids(i)  # to clear the question and the context
#
#     index = 0  # new index
#     while sequence_ids[index] != 1:
#         index += 1
#     context_start = index
#     while sequence_ids[index] == 1:
#         index += 1
#     context_end = index - 1
#     print(context_start, context_end)
#     # If the answer is not fully inside the context, label is (0,0),
#     # This (0,0) means the answer is not in this features
#     if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
#         start_positions.append(0)
#         end_positions.append(0)
#     else:
#
#         index = context_start
#         while index <= context_end and offset[index][0] <= start_char:
#             index += 1
#         start_positions.append(index - 1)
#
#         index = context_end
#         while index >= context_start and offset[index][1] >= end_char:
#             index -= 1
#         end_positions.append(index + 1)
#
# print(start_positions, end_positions)
#
# idx = 0
# features = inputs["overflow_to_sample_mapping"][0]
# start = start_positions[idx]
# end = end_positions[idx]
# answer_check = answers[features]["text"][0]
# print(answer_check)
# labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])
# print(labeled_answer)







