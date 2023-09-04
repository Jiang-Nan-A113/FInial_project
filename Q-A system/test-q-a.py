from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
The user is the robot user,  and the robot is the service provider.
In the scene, the user is between different objects. These include high-speed objects such as cars
and trucks and concrete obstacles such as tables and rocks. If the user walks towards an obstacle,
the user may be injured. If the user walks towards a high-speed object, the user may be seriously
injured. In case the user is injured, the robot will sound an alarm. If the user is seriously injured,
the robot will call the police and send an emergency text message.
"""

question_list =["What is user?",
                "What is the robot?",
                "What is the high-speed object?",
                "What are the concrete obstacles? ",
                "What will happen when a user walks towards an obstacle?",
                "What will happen when a user walks towards a high-speed object?",
                "What will happen when user is injured?",
                " What will happen when user is seriously injured?"]
i = 0
while i < 8:
    question = question_list[i]
    print(question)
    a = question_answerer(question=question, context=context)
    print(a)
    i +=1
