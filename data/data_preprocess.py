from dataset import ds

def preprocess(example):
  input_text = example['input'] if example['input'].strip() else "< noinput >"
  text = f"### Description : {example['instruction']}\n### Input: {input_text}\n### Code: {example['output']}"
  return {"text" : text}

ds = ds.map(preprocess)

train_test_data= ds['train'].train_test_split(test_size=0.2)
