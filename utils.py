import re

def construct_text_field(example):
    pattern = r'\[\|(.+?)\|\] (.+)'
    matches = re.findall(pattern, example['input'])

    human = []
    ai = []
    text_field = ""

    for speaker, message in matches:
        if speaker.lower() == "human":
            text_field += "### Human: "
            text_field += message
            text_field += " "
        else:
            text_field += "###Assistant: "
            text_field += message
            text_field += " "

    example['text'] = text_field

    return example