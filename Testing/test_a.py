#
# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id= "google/pegasus-xsum" , filename= "config.json" )
# hf_hub_download(
#       repo_id= "google/pegasus-xsum" ,
#       filename= "config.json" ,
#       revision= "4d33b01d79672f27f001f6abade33f22d993b151"
#  )
# # a = [1, 2, 3, 4, 5, 6]
# # b = a.pop()
# # print(b)
#
# import transformers
# import torch
# # from datasets import load_dataset
# # from transformers import AutoTokenizer, AlbertModel
# #
# # raw_dataset = load_dataset("squad")
# # print(raw_dataset)
# # ver = raw_dataset["validation"]
#
# from datasets import load_dataset
# from transformers import AutoTokenizer
#
# raw_datasets = load_dataset("squad")
# model_checkpoint = "bert-base-cased"  # The model is based on BERT
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#
# dataset = raw_datasets["validation"][2:6]
#
# questions = raw_datasets["validation"][2: 6]["question"]
# context = raw_datasets["validation"][2: 6]["context"]
# inputs = tokenizer(
#     questions,
#     context,
#     max_length=100,
#     truncation="only_second",
#     stride=50,
#     return_overflowing_tokens=True,
#     return_offsets_mapping=True,
#     padding="max_length",
# )
#
# sample_map = inputs["overflow_to_sample_mapping"]
# print(sample_map)
# example_ids = []
#
# print(inputs["input_ids"])
# print(len(sample_map))
# print(range(len(inputs["input_ids"])))
# print(range(len(sample_map)))
# print(inputs["offset_mapping"][0])
#
# for i in range(len(inputs["input_ids"])):
#     sample_idx = sample_map[i]
#     example_ids.append(dataset["id"][sample_idx])
#
#     sequence_ids = inputs.sequence_ids(i)
#     offset = inputs["offset_mapping"][i]
#     inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
#     print(inputs["offset_mapping"])
#
# example_1 = [0, 0, 0, 0, 1, 1, 1, 1, 1]
# example_2 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18)]
#
# example_1 = [y if example_1[x] == 1 else None for x,y in enumerate(example_2)]
# print(example_1)

# from transformers import pipeline
#
# model_checkpoint = "huggingface-course/bert-finetuned-squad"
# question_answerer = pipeline("question-answering", model=model_checkpoint)

# context = """
# 🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
# between them. It's straightforward to train your models with one before loading them for inference with the other.
# """
# question = "Which deep learning libraries back 🤗 Transformers?"
# q = question_answerer(question=question, context=context)
# print(q)

# context = """
# There are three objects in the room, a person walking to wards the object, the person might get hurt.
# """
# question = "Will the person hurt?"
# q = question_answerer(question=question, context=context)
# print(q)
# import torch
# if torch.cuda.is_available():
#     print("GPU is available.")
# else:
#     print("No GPU available, using CPU.")
#
# print(torch.__version__)
# print(torch.version.cuda)
# import collections
# import numpy as np
# from transformers import TFAutoModelForQuestionAnswering, AutoModelForQuestionAnswering
# from training import preprocess_training
# from datasets import load_dataset
# from transformers import AutoTokenizer
# from verification import preprocess_validation
#
# raw_datasets = load_dataset("squad")
# small_eval_set = raw_datasets["validation"].select(range(100))
# trained_checkpoint = "distilbert-base-cased-distilled-squad"
#
# tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
# eval_set = small_eval_set.map(
#     preprocess_validation_examples,
#     batched=True,
#     remove_columns=raw_datasets["validation"].column_names,
# )
import numpy as np
import re

output_text = """
0%|          | 0/22184 [00:00<?, ?it/s]You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
  2%|▏         | 500/22184 [01:40<1:12:54,  4.96it/s]{'loss': 2.5609, 'learning_rate': 1.9552830869094844e-05, 'epoch': 0.05}
  5%|▍         | 1000/22184 [03:28<1:13:10,  4.83it/s]{'loss': 1.5921, 'learning_rate': 1.91020555355211e-05, 'epoch': 0.09}
  7%|▋         | 1500/22184 [05:17<1:16:40,  4.50it/s]{'loss': 1.4674, 'learning_rate': 1.865128020194735e-05, 'epoch': 0.14}
  9%|▉         | 2000/22184 [07:11<1:18:50,  4.27it/s]{'loss': 1.3745, 'learning_rate': 1.8200504868373604e-05, 'epoch': 0.18}
 11%|█▏        | 2500/22184 [09:07<1:17:43,  4.22it/s]{'loss': 1.3087, 'learning_rate': 1.774972953479986e-05, 'epoch': 0.23}
 14%|█▎        | 3000/22184 [11:00<1:12:45,  4.39it/s]{'loss': 1.2576, 'learning_rate': 1.7300757302560404e-05, 'epoch': 0.27}
 16%|█▌        | 3500/22184 [12:51<1:10:34,  4.41it/s]{'loss': 1.2815, 'learning_rate': 1.6850883519653807e-05, 'epoch': 0.32}
 18%|█▊        | 4000/22184 [14:48<1:10:55,  4.27it/s]{'loss': 1.2281, 'learning_rate': 1.6400108186080058e-05, 'epoch': 0.36}
 20%|██        | 4500/22184 [16:44<1:00:46,  4.85it/s]{'loss': 1.214, 'learning_rate': 1.5949332852506312e-05, 'epoch': 0.41}
 23%|██▎       | 5000/22184 [18:36<1:04:06,  4.47it/s]{'loss': 1.147, 'learning_rate': 1.5498557518932563e-05, 'epoch': 0.45}
 25%|██▍       | 5500/22184 [20:29<1:02:27,  4.45it/s]{'loss': 1.1277, 'learning_rate': 1.504778218535882e-05, 'epoch': 0.5}
 27%|██▋       | 6000/22184 [22:23<59:02,  4.57it/s]{'loss': 1.1648, 'learning_rate': 1.4597006851785072e-05, 'epoch': 0.54}
 29%|██▉       | 6500/22184 [24:16<58:53,  4.44it/s]{'loss': 1.091, 'learning_rate': 1.4146231518211325e-05, 'epoch': 0.59}
 32%|███▏      | 7000/22184 [26:11<57:30,  4.40it/s]{'loss': 1.1458, 'learning_rate': 1.3695456184637576e-05, 'epoch': 0.63}
 34%|███▍      | 7500/22184 [28:07<58:02,  4.22it/s]{'loss': 1.1253, 'learning_rate': 1.3244680851063832e-05, 'epoch': 0.68}
 36%|███▌      | 8000/22184 [29:59<54:25,  4.34it/s]{'loss': 1.1152, 'learning_rate': 1.2793905517490083e-05, 'epoch': 0.72}
 38%|███▊      | 8500/22184 [31:51<53:22,  4.27it/s]{'loss': 1.0417, 'learning_rate': 1.2343130183916336e-05, 'epoch': 0.77}
 41%|████      | 9000/22184 [33:45<48:43,  4.51it/s]{'loss': 1.1004, 'learning_rate': 1.1892354850342592e-05, 'epoch': 0.81}
 43%|████▎     | 9500/22184 [35:38<48:22,  4.37it/s]{'loss': 1.0439, 'learning_rate': 1.1441579516768843e-05, 'epoch': 0.86}
 45%|████▌     | 10000/22184 [37:36<48:47,  4.16it/s]{'loss': 1.0443, 'learning_rate': 1.0990804183195096e-05, 'epoch': 0.9}
 47%|████▋     | 10500/22184 [39:32<47:07,  4.13it/s]{'loss': 1.0281, 'learning_rate': 1.0540028849621348e-05, 'epoch': 0.95}
 50%|████▉     | 11000/22184 [41:26<43:07,  4.32it/s]{'loss': 1.0511, 'learning_rate': 1.0089253516047603e-05, 'epoch': 0.99}
 52%|█████▏    | 11500/22184 [43:27<39:41,  4.49it/s]{'loss': 0.809, 'learning_rate': 9.639379733141003e-06, 'epoch': 1.04}
 54%|█████▍    | 12000/22184 [45:18<39:16,  4.32it/s]{'loss': 0.7683, 'learning_rate': 9.188604399567257e-06, 'epoch': 1.08}
 56%|█████▋    | 12500/22184 [47:10<33:13,  4.86it/s]{'loss': 0.7426, 'learning_rate': 8.739632167327806e-06, 'epoch': 1.13}
 59%|█████▊    | 13000/22184 [49:06<36:35,  4.18it/s]{'loss': 0.7567, 'learning_rate': 8.288856833754058e-06, 'epoch': 1.17}
 61%|██████    | 13500/22184 [51:04<34:58,  4.14it/s]{'loss': 0.7422, 'learning_rate': 7.838081500180311e-06, 'epoch': 1.22}
 63%|██████▎   | 14000/22184 [53:02<32:30,  4.20it/s]{'loss': 0.7589, 'learning_rate': 7.387306166606564e-06, 'epoch': 1.26}
 65%|██████▌   | 14500/22184 [54:59<30:18,  4.23it/s]{'loss': 0.7554, 'learning_rate': 6.936530833032817e-06, 'epoch': 1.31}
 68%|██████▊   | 15000/22184 [56:56<27:43,  4.32it/s]{'loss': 0.7665, 'learning_rate': 6.48575549945907e-06, 'epoch': 1.35}
 70%|██████▉   | 15500/22184 [58:53<26:12,  4.25it/s]{'loss': 0.7314, 'learning_rate': 6.034980165885324e-06, 'epoch': 1.4}
 72%|███████▏  | 16000/22184 [1:00:50<23:12,  4.44it/s]{'loss': 0.7683, 'learning_rate': 5.584204832311576e-06, 'epoch': 1.44}
 74%|███████▍  | 16500/22184 [1:02:48<22:01,  4.30it/s]{'loss': 0.7311, 'learning_rate': 5.13342949873783e-06, 'epoch': 1.49}
 77%|███████▋  | 17000/22184 [1:04:46<20:40,  4.18it/s]{'loss': 0.7175, 'learning_rate': 4.682654165164083e-06, 'epoch': 1.53}
 79%|███████▉  | 17500/22184 [1:06:38<17:40,  4.42it/s]{'loss': 0.748, 'learning_rate': 4.2318788315903354e-06, 'epoch': 1.58}
 81%|████████  | 18000/22184 [1:08:28<16:15,  4.29it/s]{'loss': 0.7316, 'learning_rate': 3.781103498016589e-06, 'epoch': 1.62}
 83%|████████▎ | 18500/22184 [1:10:19<12:43,  4.82it/s]{'loss': 0.7315, 'learning_rate': 3.332131265777137e-06, 'epoch': 1.67}
 86%|████████▌ | 19000/22184 [1:12:09<11:46,  4.51it/s]{'loss': 0.7209, 'learning_rate': 2.8813559322033903e-06, 'epoch': 1.71}
 88%|████████▊ | 19500/22184 [1:14:02<10:34,  4.23it/s]{'loss': 0.7166, 'learning_rate': 2.4305805986296434e-06, 'epoch': 1.76}
 90%|█████████ | 20000/22184 [1:15:52<08:15,  4.41it/s]{'loss': 0.7245, 'learning_rate': 1.979805265055896e-06, 'epoch': 1.8}
 92%|█████████▏| 20500/22184 [1:17:43<05:45,  4.88it/s]{'loss': 0.7435, 'learning_rate': 1.5290299314821493e-06, 'epoch': 1.85}
 95%|█████████▍| 21000/22184 [1:19:33<04:26,  4.44it/s]{'loss': 0.7063, 'learning_rate': 1.07915614857555e-06, 'epoch': 1.89}
 97%|█████████▋| 21500/22184 [1:21:23<02:32,  4.49it/s]{'loss': 0.7153, 'learning_rate': 6.301839163360981e-07, 'epoch': 1.94}
 99%|█████████▉| 22000/22184 [1:23:13<00:41,  4.40it/s]{'loss': 0.72, 'learning_rate': 1.7940858276235126e-07, 'epoch': 1.98}
100%|██████████| 22184/22184 [1:24:20<00:00,  4.38it/s]帮我提取'learning_rate'
"""

# 从输出文本中提取'learning_rate'的值
learning_rates = [float(match.group(1)) for match in re.finditer(r"'learning_rate': ([\d.e-]+)", output_text)]

# 将提取的值存储为一维向量
learning_rates = np.array(learning_rates)

# 打印前几个值
print(learning_rates[:10])  # 打印前10个learning_rate的值






