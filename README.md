# Mamba SSM Code Generation Model

A from-scratch implementation of the **Mamba State Space Model (SSM)** for code generation tasks. This project implements a language model based on the Mamba architecture to generate Python code from natural language descriptions.

## 🎯 Project Overview

This project aims to build a production-ready code generation model using the Mamba architecture, a modern alternative to Transformers that offers:
- **Linear time complexity** in sequence length
- **Efficient memory usage** 
- **Superior performance** on long sequences

The model is trained on the **CodeAlpaca-20k** dataset and can generate Python code based on task descriptions and optional input parameters.

## 📊 Training Results

- **Training Loss**: ~0.30
- **Validation Loss**: ~0.49
- **Model Checkpoint**: epoch 10
- **Hyperparameter Tuning**: Tracked with Weights & Biases (W&B)

### Training Configuration
```
- Learning Rate: 0.0005
- Batch Size: 8
- Epochs: 20
- Dropout Rate: 0.3
- Vocabulary Size: 50,257 (GPT-2 tokenizer)
- Model Dimension (d_model): 256
- Number of Layers: 6
- Max Sequence Length: 512
```

## 📁 Project Structure

```
mamba_codeGen/
├── config.py                          # Global configuration and model initialization
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
│
├── model/
│   ├── __init__.py
│   ├── mamba_block.py                # Core Mamba block implementation
│   │   ├── Layer normalization
│   │   ├── State space transitions (A, B, C matrices)
│   │   ├── Cumulative product & sum operations
│   │   └── Residual connections with dropout
│   └── mamba_lm.py                   # Language model wrapper
│       ├── Token embedding layer
│       ├── Stack of Mamba blocks
│       └── Output projection layer
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                    # Dataset loading from Hugging Face
│   ├── data_preprocess.py            # Data preprocessing & formatting
│   ├── tokenized_data.py             # Tokenization using GPT-2 tokenizer
│   └── dataLoader.py                 # DataLoader for training
│
├── training/
│   ├── __init__.py
│   ├── train.py                      # Training loop with W&B integration
│   └── dataLoader.py                 # Batch preparation & data loading
│
├── inference/
│   ├── __init__.py
│   └── generate.py                   # Code generation script
│
├── api/
│   ├── __init__.py
│   └── app.py                        # (Placeholder for REST API)
│
└── artifacts/
    └── mamba_checkpoint_epoch10.pt   # Trained model checkpoint
```

## 🏗️ Architecture Overview

### Mamba Block
The core component of our model implements the Mamba SSM mechanism:

```
Input → LayerNorm → A, B, C Projections
              ↓
         State Space → Cumulative Operations
              ↓
         Output → Dropout → Residual Connection
```

**Key Operations:**
- **A matrix**: Controls state transitions via sigmoid
- **B matrix**: Input projection for state space
- **C matrix**: Output projection from hidden state
- **Cumulative product/sum**: Efficient selective scan computation

### MambaLM
The language model combines:
1. **Token Embedding** (vocab_size → d_model)
2. **6 Stacked Mamba Blocks** (sequence processing)
3. **Output Projection** (d_model → vocab_size)

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd mamba_codeGen
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets wandb
   ```

## 📈 Training with W&B

The training process integrates **Weights & Biases** for:
- **Real-time metric tracking** (training/validation loss)
- **Hyperparameter logging** (learning rate, batch size, etc.)
- **Model checkpointing** (automatic best model saving)
- **Experiment comparison** (across different runs)

**Training Command:**
```bash
cd training
python train.py
```

Monitor your training in real-time:
```
https://wandb.ai/your-username/mamba-code-gen
```

## 💻 Running Inference

### Quick Start - Generate Code

```bash
python -m inference.generate
```

This will generate a Python factorial function using the trained model.

### Example Usage

```python
from inference.generate import generate, promptFormat
from config import device

# Format your prompt
prompt = promptFormat(
    instruction="Write a Python function to reverse a list",
    input="[1, 2, 3, 4, 5]"
)

# Generate code
generated_code = generate(prompt, max_new_tokens=150)
print(generated_code)
```

### Prompt Format

The model expects prompts in this format:
```
### Description : <task_description>
### Input: <input_data>
### Code:
```

**Examples:**
```
### Description : Create an array of length 5 with even numbers between 1 and 10
### Input: < noinput >
### Code:
```

## 🔧 Configuration

Edit `config.py` to customize model parameters:

```python
vocab_size = 50257              # GPT-2 vocabulary
d_model = 256                   # Hidden dimension
n_layers = 6                    # Number of Mamba blocks
dropout_rate = 0.3             # Dropout probability
epochs = 20                     # Training epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Mamba SSM |
| Tokenizer | GPT-2 |
| Vocabulary Size | 50,257 |
| Embedding Dimension | 256 |
| Number of Layers | 6 |
| Dropout Rate | 0.3 |
| Training Dataset | CodeAlpaca-20k |
| Dataset Size | 16,017 examples |
| Max Sequence Length | 512 tokens |

## 🎓 Key Features

✅ **State Space Model**: Linear-time complexity alternative to Transformers  
✅ **Efficient Architecture**: Lower memory footprint than attention-based models  
✅ **Code Generation**: Trained on CodeAlpaca dataset for Python code generation  
✅ **Hyperparameter Tuning**: Full W&B integration for experiment tracking  
✅ **Modular Design**: Clean separation of concerns (model, data, training, inference)  
✅ **GPU Acceleration**: CUDA support for fast training and inference  

## 📚 Dataset

- **Source**: [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- **Size**: 16,017 examples
- **Format**: JSON with instruction, input, and code output
- **Processing**: Automatic download and preprocessing via Hugging Face Datasets

## 🤝 Dependencies

- `torch` - Deep learning framework
- `transformers` - GPT-2 tokenizer and utilities
- `datasets` - Dataset loading and preprocessing
- `wandb` - Experiment tracking and hyperparameter tuning
- `numpy` - Numerical operations

## 📝 Loss Curves

The model achieved good convergence during training:
- **Training Loss**: Started high, converged to ~0.30
- **Validation Loss**: ~0.49 (indicating slight overfitting, manageable with regularization)
- **Training Stability**: Smooth loss decrease over 20 epochs

## 🔮 Future Improvements

- [ ] Implement beam search for better code generation quality
- [ ] Add temperature and top-k sampling strategies
- [ ] Deploy REST API for inference
- [ ] Model quantization for edge deployment
- [ ] Extended context length (> 512 tokens)

## 🐛 Troubleshooting

**Issue**: CUDA out of memory
- Solution: Reduce batch size in `config.py` or use CPU

**Issue**: Slow data loading
- Solution: First run downloads dataset - subsequent runs are cached

**Issue**: Import errors
- Solution: Use relative imports from package root: `python -m inference.generate`

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Mamba Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" by Gu & Dao
- **CodeAlpaca Dataset**: Sahil2801 on Hugging Face Hub
- **GPT-2 Tokenizer**: OpenAI

---

**Created**: April 2026  
**Status**: - Ready for inference and experimentation , But further Training is ongoing, on a Bigger Dataset for Quality responses.
