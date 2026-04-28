
from pyexpat import model

import torch
import wandb
import torch.nn as nn
from training.dataLoader import train_data, test_data
import config

device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
config.model.to(device)
wandb.init(
    project="mamba-code-gen",
    config={
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "epochs": config.epochs,
        "max_length": 256,
        "batch_size": 8
    }
)
global_step=0
criterion = nn.CrossEntropyLoss()
def training_step(batch):
    config.optimizer.zero_grad()
    input_ids = torch.tensor(batch['input_ids'].to(device))
    X_in = input_ids[:,:-1]
    targets = input_ids[:,1:]
    logits = config.model(X_in)
    B,T,V = logits.shape
    loss = criterion(logits.reshape(B*T,V), targets.reshape(B*T))
    loss.backward()
    config.optimizer.step()
    return loss.item()

def validation_step(batch):
    with torch.no_grad():
        input_ids = torch.tensor(batch['input_ids'].to(device))
        X_in = input_ids[:,:-1]
        targets = input_ids[:,1:]
        logits = model(X_in)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss.item()


for epoch in range(config.epochs):
    model.train()
    train_losses=[]
    for step, batch in enumerate(train_data):
        train_loss = training_step(batch)
        train_losses.append(train_loss)
        global_step +=1
        wandb.log({"train_loss": train_loss, "step": global_step})

    if epoch%1==0:
        model.eval()
        val_losses =[]
        for step, batch in enumerate(test_data):
            val_loss = validation_step(batch)
            val_losses.append(val_loss)
        wandb.log({"val_loss": sum(val_losses)/len(val_losses), "epoch": epoch})
        print(f"Validation loss for epoch {epoch} : {sum(val_losses)/len(val_losses)}")

    if epoch %1 == 0 :
        print(f"epoch:{epoch} / training loss : {sum(train_losses)/len(train_losses)}")
        torch.save({
            'model_state_dict': config.model.state_dict(),
            'optimizer_state_dict': config.optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': sum(train_losses)/len(train_losses)
        }, f"/content/drive/MyDrive/mamba_checkpoint_epoch{epoch}.pt")


wandb.finish()
