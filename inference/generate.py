import torch 
from config import model
from config import device
from data.tokenized_data import tokenizer

checkpoint = torch.load("artifacts/mamba_checkpoint_epoch10.pt", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

def generate(prompt, max_new_tokens=100):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    X = torch.tensor(token_ids).unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(X)                        # (1, T, vocab_size)
            next_token_logits = logits[0, -1, :]     # last position
            next_token = torch.argmax(next_token_logits)  # pick highest prob
            X = torch.cat([X, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    fullToken = tokenizer.decode(X[0].tolist())
    output_token = fullToken.split("### Code:")[-1].strip()
    return output_token

def promptFormat(instruction,input="< noinput >"):
    return f"### Description : {instruction}\n### Input: {input}\n### Code:"
    
prompt = promptFormat("Write a Python function to calculate the factorial of a number.", "5")
generated_code = generate(prompt)
print(generated_code)