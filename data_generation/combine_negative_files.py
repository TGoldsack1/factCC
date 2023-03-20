import json


def my_load_data(path):

    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    return data


data_pronoun = my_load_data("/home/acp20tg/factCC/data_generation/raw_data/st1_PLOS_formatted-pronoun.jsonl")
data_dateswp = my_load_data("/home/acp20tg/factCC/data_generation/raw_data/st1_PLOS_formatted-dateswp.jsonl")
data_numswp = my_load_data("/home/acp20tg/factCC/data_generation/raw_data/st1_PLOS_formatted-numswp.jsonl")
data_entswp = my_load_data("/home/acp20tg/factCC/data_generation/raw_data/st1_PLOS_formatted-entswp.jsonl")
data_negation = my_load_data("/home/acp20tg/factCC/data_generation/raw_data/st1_PLOS_formatted-negation.jsonl")


data_negative = data_pronoun + data_dateswp + data_numswp + data_entswp + data_negation

def save_data(data, name_suffix):
    output_file = "/home/acp20tg/factCC/data_generation/raw_data/st1_PLOS_formatted-negative.jsonl"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            # example = dict(example)
            # example["text"] = example["text"].text
            # example["claim"] = example["claim"].text
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")



save_data(data_negative, "negative")
