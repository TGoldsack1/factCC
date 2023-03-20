import json, random


subtask = "st2_PLOS"

source_dir = "/fastdata/acp20tg/factcc_data"

data_files = [
    f"{source_dir}/{subtask}_formatted-negative.jsonl",
    f"{source_dir}/{subtask}_formatted-positive.jsonl",
    f"{source_dir}/{subtask}_formatted-negative-noise.jsonl",
    f"{source_dir}/{subtask}_formatted-positive-noise.jsonl"
]

def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

t = 0
all_data = []
for f in data_files:
    data = load_source_docs(f)
    all_data += data
    print(len(data))
    t += len(data)

print(t)

random.shuffle(all_data)

# get 5% data for dev
dev_data = all_data[:int(len(all_data)*0.05)]
train_data = all_data[int(len(all_data)*0.05):]

# write to file
with open(f"{source_dir}/data-train.jsonl", "w") as out_f:
    for d in train_data:
        out_f.write(json.dumps(d) + "\n")

with open(f"{source_dir}/data-dev.jsonl", "w") as out_f:
    for d in dev_data:
        out_f.write(json.dumps(d) + "\n")
