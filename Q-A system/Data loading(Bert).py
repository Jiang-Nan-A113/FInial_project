from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("squad")
# print(raw_datasets["validation"][0])

# print("Id: ", raw_datasets["validation"][0]["id"])
# print("Title: ", raw_datasets["validation"][0]["title"])
# print("Context: ", raw_datasets["validation"][0]["context"])
# print("Question: ", raw_datasets["validation"][0]["question"])
# print("Answer: ", raw_datasets["validation"][0]["answers"])

# raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
#   Filtering text with more than one answer

# print(raw_datasets["validation"][0]["answers"])
# print(raw_datasets["validation"][2]["answers"])
# print(raw_datasets["validation"][2]["context"])
# print(raw_datasets["validation"][2]["question"])

model_checkpoint = "bert-base-cased"  # The model is based on BERT
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("This tokenizer is fast" if tokenizer.is_fast is True else "error")  # Check if the Tokenizer is 'fast' version

# context = raw_datasets["train"][0]["context"]
# question = raw_datasets["train"][0]["question"]

# inputs = tokenizer(question, context)   # The Tokenizer will insert the special marker automatically
# # print(tokenizer.decode(inputs["input_ids"]))  # Examples of insert the special marker

# inputs = tokenizer(
#     question,
#     context,
#     max_length=100,
#     truncation="only_second",
#     stride=50,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
# )   # If the input: (question and context) longer than 100, the tokenizer will follow the truncation set and return
# # a mark
#
# print(inputs.keys)

# for ids in inputs["input_ids"]:
#     print(tokenizer.decode(ids))    # Output to the human read-able version

# inputs = tokenizer(
#     raw_datasets["train"][2:6]["question"],
#     raw_datasets["train"][2:6]["context"],
#     # The 2 to 5, because the index is from zero, include start but without ending.
#     max_length=100,
#     truncation="only_second",
#     stride=50,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
# )
# # print(inputs.keys)
# # print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
# # print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
# #   Here is where each comes from: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3].
# #   The example shows the features are from 4 in 0, 4 in 1, 4 in 2 and 7 in 3.
#
# answers = raw_datasets["train"][2: 6]["answers"]
# # print("offset_mapping", inputs["offset_mapping"])
# # print("overflow_to_sample_mapping", inputs["overflow_to_sample_mapping"][0])
# start_positions = []
# end_positions = []
# """
# The features totally are 17, the index of the feature is not change:
# [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3].
# So the answer for the same part is same.
# """
# for i, offset in enumerate(inputs["offset_mapping"]):    # The sentence over 100(start position: end position)
#     sample_idx = inputs["overflow_to_sample_mapping"][i]
#     answer = answers[sample_idx]
#     answer_start_char = answer["answer_start"][0]  # Get the start chat of the answer
#     answer_end_char = answer["answer_start"][0] + len(answer["text"][0])   # Get the end chat of the answer
#     sequence_ids = inputs.sequence_ids(i)
#     # print(sequence_ids)
#     # To distinguish the context and answer, 0 is question and 1 is context、
#
#     idx = 0
#     while sequence_ids[idx] != 1:   # Judge whether sequence_ids[idx] not equal to 1
#         # print(idx)
#         # print(sequence_ids[idx])
#         idx += 1
#         # print(idx)
#     context_start = idx # The end of question means the start of context
#     # print(context_start)
#     while sequence_ids[idx] == 1:
#         # print(sequence_ids[idx])
#         idx += 1
#     context_end = idx - 1
#     # print(context_end)
#     # print('offset', offset[context_start][0])
#     # print('start_char', answer_start_char)
#     if offset[context_start][0] > answer_start_char or offset[context_end][1] < answer_end_char:
#         start_positions.append(0)
#         end_positions.append(0)
#     else:
#         idx = context_start
#         # print(idx)
#         while idx <= context_end and offset[idx][0] <= answer_start_char:
#             idx += 1
#             # print(idx)
#         start_positions.append(idx - 1)
#         # print(start_positions)
#
#         idx = context_end
#         while idx >= context_start and offset[idx][1] >= answer_end_char:
#             idx -= 1
#         end_positions.append(idx + 1)

# print(start_positions, end_positions)   # Find the answer start and end
"""
The features is: 
[None, 0, 0, 0, 0, 0,... 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, ... 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
so use the 0 and 1, we can have the start_position and enc_position of question and context
"""


inputs_v = tokenizer(
    raw_datasets["train"]["question"][0: 3],
    raw_datasets["train"]["context"][0: 3],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)

sample_map = inputs_v.pop("overflow_to_sample_mapping")
example_ids = []
dataset = raw_datasets["validation"]
print(dataset)
for i in range(len(inputs_v["input_ids"])):
    sample_idx = sample_map[i]
    print(sample_idx)
    b = example_ids.append(dataset["id"][sample_idx])
    print('example_ids', b)
    sequence_ids = inputs_v.sequence_ids(i)
    # print(sequence_ids)
    offset = inputs_v["offset_mapping"][i]
    # print("offset_mapping", offset)
    a = inputs_v["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
    # print(a)
# offset_mapping = inputs_v["offset_mapping"]
# answers = raw_datasets["train"][2: 6]["answers"]
# # print(answers_v)
#
# start_char = []
# end_char = []
#
# for i, offset_v in enumerate(offset_mapping):
#     sample_maps = inputs_v["overflow_to_sample_mapping"]
#     sample_index = sample_maps[i]  # Find the features is from which sentences
#     answer_v = answers[sample_index]  # Find the answers of the features
#     # print(answer_v)
#     start_answer = answer_v['answer_start'][0]
#     end_answer = answer_v['answer_start'][0] + len(answer_v['text'][0])
#     sequence_ids = inputs_v.sequence_ids(i)
#     print(sequence_ids)
#
#     index = 0
#     while sequence_ids[index] != 1:
#         index += 1
#     context_start = index
#     while sequence_ids[index] == 1:
#         index += 1
#     context_end = index - 1
#     # print(context_start, context_end)
#     # print(offset_v[context_start][0], offset_v[context_end][1])
#     if offset_v[context_start][0] > end_answer or offset_v[context_end][1] < start_answer:
#         start_char.append(0)
#         end_char.append(0)
#     else:
#         index = context_start
#         while index <= context_end and offset_v[index][0] <= start_answer:
#             index += 1
#         start_char.append(index - 1)
#         index = context_end
#         while index >= context_start and offset_v[index][1] >= end_answer:
#             index -= 1
#         end_char.append(index + 1)
# print(start_char, end_char)





