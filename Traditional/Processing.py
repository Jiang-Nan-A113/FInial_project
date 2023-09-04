def subject_find(tag):
    subject_str = []
    tag_sent = [x for x, y in enumerate(tag) if y[1] == "NN" and "NNS"]
    # print(tag_sent)
    subject_sent = tag[tag_sent[-1]]
    for x in subject_sent:
        subject_str.append(x)
    return subject_str


def object_find(tag_words):
    object_str = []
    tag_sent = ""
    prep_bool = bool([x for x, y in enumerate(tag_words) if y[1] == "TO"])
    # print(prep_bool)
    if prep_bool is True:
        tag_sent = [x for x, y in enumerate(tag_words) if y[1] == "NN" and "NNS"]
    elif prep_bool is False:
        tag_sent = [x for x, y in enumerate(tag_words) if y[1] == "PRP"]
    # print(tag_sent)
    object_sent = tag_words[tag_sent[0]]
    for x in object_sent:
        object_str.append(x)
    return object_str


def action_find(tag):
    action_str = []
    tag_sent = [x for x, y in enumerate(tag) if y[1] == "VBG" and "VB"]
    if not tag_sent:
        tag_sent = [x for x, y in enumerate(tag) if y[1] == "VBD"]
    tag_sent = tag[tag_sent[0]]
    for x in tag_sent:
        action_str.append(x)
    return action_str


def direction_find(tag):
    direction_str = ''
    toward_list = ["toward, to"]
    toward_bool = ""
    away_list = ["away, from"]
    away_bool = ""
    other_list = ["next"]
    other_bool = ""

    toward = [x for x, y in enumerate(tag) if y[1] == "IN"]
    if toward:
        toward_bool = bool([x for x in toward_list if x == toward[-1]])
        direction_str = 0
        return direction_str

    away = [x for x, y in enumerate(tag) if y[1] == "RB"]
    if away:
        away_bool = bool([x for x in away_list if x == away[-1]])
        direction_str = 1
        return direction_str

    other = [x for x, y in enumerate(tag) if y[1] == "JJ"]
    if other:
        away_bool = bool([x for x in other_list if x == other[-1]])
        direction_str = 2
        return direction_str
    # direction_0 = [x for x, y in enumerate(tag) if y[1] == "IN"]
    # if direction_0:
    #     direction_str = 1
    print(toward_bool)

    # direction_1 = [x for x, y in enumerate(tag) if y[1] == "RB"]
    # if direction_1:
    #     direction_str = 2
    # # print(direction_1)

    # direction_2 = [x for x, y in enumerate(tag) if y[1] == "JJ"]
    # if direction_2:
    #     direction_str = 3
    # print(direction_2)


def amount_find(tag):
    amount_str = ""
    amount = [x for x, y in enumerate(tag) if y[1] == 'CD']
    if amount:
        amount = tag[amount[0]]
    if not amount:
        amount_list = ['group', 'multiple', 'pair', 'multiple', 'load']
        amount = bool([x for x, y in enumerate(tag) if (y[0] in amount_list)])
        if amount is True:
            amount = ['multiple']
        elif amount is False:
            amount = ['singular']
    amount_str = amount[0]
    return amount_str


def perspective_find(tag):
    situation_focus = ""
    perspective_str = []
    perspective = [x for x, y in enumerate(tag) if y[1] == 'VB']
    perspective = tag[perspective[0]]
    if perspective[0] == 'happen':
        situation_focus = 'Open situation'
    else:
        perspective = [x for x, y in enumerate(tag) if y[1] == "PRP"]
        perspective = tag[perspective[0]]
        if perspective[0] == 'you':
            situation_focus = 'The robot'
    return situation_focus


def danger_find(tag):
    none1 = [x for x, y in enumerate(tag) if y[1] == 'NN']
    none1 = [y[0] for x, y in enumerate(tag)]
    return none1


