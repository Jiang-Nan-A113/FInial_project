import time
from http import client
from twilio.rest import Client
from geopy.geocoders import Nominatim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class Sentence:
    def __init__(self, subject_inp, predicate_inp, object_inp):
        self.subject_inp = subject_inp
        self.predicate_inp = predicate_inp
        self.object_inp = object_inp


def danger_judge(level, ):
    message = []
    if level == 'top':
        lat, long = geography()
        send_msg("This is an automatic message from the system, "
                 " the user is in an emergency situation, "
                 "level 4 (most dangerous). Requesting emergency assistance from the relevant unit."
                 "The user location is {lag} and {long}")
        call_num(999)
        message.append("level 4")
    elif level == 'dangerous':
        message.append("level 3")
    elif level == 'vigilant':
        message.append("level 2")
    elif level == 'safe':
        message.append("level 1")
    return message


def send_msg(message):
    account_sid = 'AC824d4762e6157222df45923b7298b8e9'
    auth_token = '1ce1a1cf61ce4005c1340b86148fc8f4'
    client = Client(account_sid, auth_token)
    msg = client.messages.create(
        to='+8618013526121',
        from_="+447897013854",
        body=message
    )
    return msg


def call_num(number):
    call = client.calls.create(
        to='+4407594394774',
        from_='+447897013854',
        url="http://demo.twilio.com/docs/voice.xml"
    )


def judge(situation, subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool, none1):
    if situation == 'Open situation':
        open_situation(subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool)
    elif situation == 'The robot':
        the_robot(subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool, none1)
    else:
        print('The question is out of range, please check')
    return None


def geography():
    geolocator = Nominatim(user_agent="MyApp")
    location = geolocator.geocode("Hyderabad")
    lat = location.latitude
    long = location.longitude
    print("The latitude of the location is: ", location.latitude)
    print("The longitude of the location is: ", location.longitude)
    return lat, long


def open_situation(subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool):
    action = action_sentence
    action_list = ["standing"]
    amount = amount_sentence
    if direction_bool == 0:
        level = danger_judge('dangerous')
        sentence = Sentence('robot', 'will remind', subject_sentence[0])
        print(
            f"The {sentence.subject_inp} {sentence.predicate_inp} user away from {sentence.object_inp}, risk level: {level}")
    elif direction_bool == 1:
        level = danger_judge('vigilant')
        sentence = Sentence('robot', 'will monitor', 'danger level')
        print(f"The {sentence.subject_inp} {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    elif direction_bool == 2:
        level = danger_judge('vigilant')
        sentence = Sentence('robot', 'will monitor', 'danger level')
        print(f"The {sentence.subject_inp} the {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    else:
        action_bool = bool([x for x in action if x in action_list])
        if action_bool:
            level = danger_judge('vigilant')
        sentence = Sentence('robot', 'will monitor', 'danger level')
        print(f"The {sentence.subject_inp} the {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")


def the_robot(subject_sentence, object_sentence, action_sentence, amount_sentence, direction_bool, none1):
    subject_none = subject_sentence
    object_none = object_sentence
    action = action_sentence
    amount = amount_sentence
    danger = none1
    low_speed_list = ["moving", "walking", ]
    high_speed_list = ["running"]
    no_speed_list = ["standing"]
    danger_list = ["car", "truck"]

    if direction_bool == 0:
        bool_1 = bool([x for x in action if x in low_speed_list])
        bool_2 = bool([x for x in action if x in high_speed_list])
        bool_3 = bool([x for x in action if x in no_speed_list])
        bool_4 = bool([x for x in danger if x in danger_list])
        print(bool_4)
        if bool_4:
            level = danger_judge('top')
            sentence = Sentence('risk', 'in top level', 'danger level')
            print(
                f"The robot detected the high speed object, risk level: {level}")
            print(
                f"The {sentence.subject_inp} {sentence.predicate_inp}, high speed object detected, risk level: {level} "
            )
        else:
            if bool_1:
                level = danger_judge('dangerous')
                sentence = Sentence('robot', 'will remind', subject_sentence[0])
                print(
                    f"The {sentence.subject_inp} {sentence.predicate_inp} user away from {sentence.object_inp}, risk level: {level}")
            elif bool_2:
                level = danger_judge('dangerous')
                sentence = Sentence('robot', 'will remind', subject_sentence[0])
                print(
                    f"The {sentence.subject_inp} the {sentence.predicate_inp} user away from {sentence.object_inp}, risk level: {level}")
            elif bool_3:
                level = danger_judge('vigilant')
                sentence = Sentence('robot', 'will monitor', 'danger level')
                print(f"The {sentence.subject_inp} the {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    elif direction_bool == 1:
        bool_1 = bool([x for x in action if x in low_speed_list])
        bool_2 = bool([x for x in action if x in high_speed_list])
        bool_3 = bool([x for x in action if x in no_speed_list])
        bool_4 = bool([x for x in action if x in danger_list])
        level = danger_judge('vigilant')
        sentence = Sentence('robot', 'will monitor', 'danger level')
        print(f"The {sentence.subject_inp} the {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")

    elif direction_bool == 2:
        bool_1 = bool([x for x in action if x in low_speed_list])
        bool_2 = bool([x for x in action if x in high_speed_list])
        bool_3 = bool([x for x in action if x in no_speed_list])
        bool_4 = bool([x for x in action if x in danger_list])
        level = danger_judge('vigilant')
        sentence = Sentence('robot', 'will monitor', 'danger level')
        print(f"The {sentence.subject_inp} the {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    else:
        level = danger_judge('dangerous')
        sentence = Sentence('robot', 'will remind', subject_sentence[0])
        print(
            f"The {sentence.subject_inp} {sentence.predicate_inp} user away from {sentence.object_inp}, risk level: {level}")

    # if action == "stand":
    #     level = danger_judge('vigilant')
    #     sentence = Sentence('robot', 'will monitor', 'danger level')
    #     print(
    #         f"The {sentence.subject_inp} {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    # none1 = bool([x for x, y in enumerate(none1) if y[0] == 'car'])
    # if none1 is True:
    #     if direction_bool == 1:
    #         level = danger_judge('top')
    #         sentence = Sentence('robot', 'will remind', 'danger level')
    #         print(
    #             f"The {sentence.subject_inp} {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    #     elif direction_bool == 2:
    #         level = danger_judge('top')
    #         sentence = Sentence('robot', 'will remind', 'danger level')
    #         print(
    #             f"The {sentence.subject_inp} {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")
    # if direction_bool == 1:
    #     level = danger_judge('dangerous')
    #     sentence = Sentence('robot', 'will remind', object_sentence)
    #     print(
    #         f"The {sentence.subject_inp} {sentence.predicate_inp} user away from {object_sentence}, risk level: {level}")
    # elif direction_bool == 2:
    #     level = danger_judge('vigilant')
    #     sentence = Sentence('robot', 'will monitor', 'danger level')
    #     print(f"The {sentence.subject_inp} {sentence.predicate_inp}{sentence.object_inp}, risk level: {level} ")


