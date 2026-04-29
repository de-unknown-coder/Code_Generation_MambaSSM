from .data_preprocess import train_test_data
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
  return tokenizer(
      example['text'],
      truncation=True,
      max_length=512,
      padding='max_length'
  )

tokenized_trainData = train_test_data['train'].map(tokenize)
tokenized_testData = train_test_data['test'].map(tokenize)

tokenized_trainData = tokenized_trainData.remove_columns(['instruction', 'input', 'output', 'text'])
tokenized_testData = tokenized_testData.remove_columns(['instruction', 'input', 'output', 'text'])