from datasets import load_dataset

ds = load_dataset("sahil2801/CodeAlpaca-20k")
print(ds['train'][0])


