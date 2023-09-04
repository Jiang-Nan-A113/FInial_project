import nltk
import Processing
# import Answering
from http import client
from twilio.rest import Client
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Processing import subject_find
from Processing import object_find
from Processing import action_find
from Processing import direction_find
from Processing import amount_find
from Processing import perspective_find
from Processing import danger_find
from Answering import judge


class InfoInput:
    def __init__(self, name, age, call):
        self.name = name
        self.age = age
        self.call = call
        print(f'Information：[user name：{name}，age：{age}，emergency call：{call}.]')


def input_info():
    name = input("Input the user name：")
    age = input("Input the user age：")
    call = input("Input the user emergency call：")
    return name, age, call


question_list = ["What do you do if someone were walking toward an obstacle?",
                 "What do you do if  multiple people were walking toward an obstacle?",
                 "What would you do if  5 people were moving toward an obstacle?",
                 "What would you do if  someone were standing next to an obstacle?",
                 "What would you do if  someone were running away from an obstacle?",
                 "What would you do if  a car were running toward from an obstacle?",
                 "What would you do if 2 people in a car were moving toward an obstacle?",
                 "What would you do if someone were moving toward a person?",
                 "What would you do if someone were moving toward a hole?",
                 "What would you do if someone were standing?",
                 "What would you do if walking toward an obstacle?",

                 "What would happen to someone if an obstacle were being walked toward by a person?",
                 "What would happen to someone if someone were walking toward an obstacle?",
                 "What would happen to someone if they were walking toward an obstacle?",
                 "What would happen to the obstacle if someone were walking toward the obstacle?",
                 "What would happen to someone if someone were walking toward an obstacle?"]
i = 0
stopwords = set(stopwords.words('english'))
# name, age, cell = input_info()
question = "What would you do if  a car were running toward from an obstacle?"
participle = RegexpTokenizer(r'\w+')
words = participle.tokenize(question)
words = [x.lower() for x in words]
words_tag = nltk.pos_tag(words)
# print("words_tag: ", words_tag)
words_stop = [x for x in words if not x.lower() in stopwords]
tagged_sent = nltk.pos_tag(words_stop)
# print("tagged_sent: ", tagged_sent)

subject_sentence = subject_find(tagged_sent)
print('subject_find: ',subject_sentence)
object_sentence = object_find(words_tag)
print('object_find:', object_sentence)
action_sentence = action_find(tagged_sent)
print('action_find:' , action_sentence)
amount_sentence = amount_find(tagged_sent)
print('amount_find:', amount_sentence)
none1 = danger_find(words_tag)
print('none1:', none1)
situation = perspective_find(words_tag)
print("perspective_find:", situation)
direction_bool = direction_find(tagged_sent)
print("direction_find:", direction_bool)
judge(situation, subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool, none1,)

# stopwords = set(stopwords.words('english'))
# question = question_list[num]
# participle = RegexpTokenizer(r'\w+')
# words = participle.tokenize(question)
# words = [x.lower() for x in words]
# words_tag = nltk.pos_tag(words)
# print("words_tag: ", words_tag)
# words_stop = [x for x in words if not x.lower() in stopwords]
# tagged_sent = nltk.pos_tag(words_stop)
# print("tagged_sent: ", tagged_sent)
#
# subject_sentence = subject_find(tagged_sent)
# # print(subject_sentence)
# object_sentence = object_find(words_tag)
# print(object_sentence)
# action_sentence = action_find(tagged_sent)
# print(action_sentence)
# amount_sentence = amount_find(tagged_sent)
# situation = perspective_find(words_tag)
# none1 = danger_find(words_tag)
# print('subject is: %s '
#       'object is: %s '
#       'action is %s' % (subject_sentence[0], object_sentence[0], action_sentence[0]))
# print(r'direction_bool is : %s' % direction_bool)
# print(r'amount_sentence is : %s' % amount_sentence)
# print(r'situation is : %s' % situation)
#
# judge(situation, subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool, none1, none2)
