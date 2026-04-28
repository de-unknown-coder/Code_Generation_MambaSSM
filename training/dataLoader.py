from data.tokenized_data import tokenized_trainData, tokenized_testData
from torch.utils.data import DataLoader

tokenized_trainData.set_format("torch")
tokenized_testData.set_format("torch")
train_data = DataLoader( tokenized_trainData , shuffle=True, batch_size=8)
test_data = DataLoader(tokenized_testData, shuffle = False , batch_size=8)
