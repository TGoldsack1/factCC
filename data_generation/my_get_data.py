import json

# Paths to the data
st1_eLife_path = "/home/tomasg/Code/bionlp-shared_task/data/final_task_data/task1-ls/train/eLife_train.jsonl"
st1_PLOS_path = "/home/tomasg/Code/bionlp-shared_task/data/final_task_data/task1-ls/train/PLOS_train.jsonl"
st2_PLOS_path = "/home/tomasg/Code/bionlp-shared_task/data/final_task_data/task2-rc/train.jsonl"

# Get the data
def get_data(path):
    with open(path) as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    return data

st1_eLife = get_data(st1_eLife_path)
st1_PLOS = get_data(st1_PLOS_path)
st2_PLOS = get_data(st2_PLOS_path)


# Get the data in format: { "id": str, "text": str }
def get_data_formatted(data):
    return [{"id": d["id"], "text": d["article"]} for d in data]

st1_eLife_formatted = get_data_formatted(st1_eLife)
st1_PLOS_formatted = get_data_formatted(st1_PLOS)
st2_PLOS_formatted = get_data_formatted(st2_PLOS)


# total_chars = 0
print(sum([len(d["text"]) for d in st1_eLife_formatted]))
print(sum([len(d["text"]) for d in st1_PLOS_formatted]))
print(sum([len(d["text"]) for d in st2_PLOS_formatted]))
# print(total_chars)


# Save the data
def save_data(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")



save_data(st1_eLife_formatted, "./raw_data/st1_eLife_formatted.jsonl")
save_data(st1_PLOS_formatted, "./raw_data/st1_PLOS_formatted.jsonl")
save_data(st2_PLOS_formatted, "./raw_data/st2_PLOS_formatted.jsonl")
