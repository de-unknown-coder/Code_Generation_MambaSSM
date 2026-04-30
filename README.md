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

## 🚀 Quick Start

### Installation

1. **Clone or download the repository**:
```bash
cd mamba_codeGen
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch` - PyTorch for model computation
- `transformers` - Hugging Face library for tokenizers and datasets
- `fastapi` - REST API framework
- `uvicorn` - ASGI server for running FastAPI
- `pydantic` - Data validation for API requests

### Running the API

Start the FastAPI server:
```bash
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- **API Documentation**: Visit `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: Visit `http://localhost:8000/redoc` (ReDoc)

## 📡 API Endpoints

### `POST /generate`

Generate Python code based on task description and optional input.

**Request Body:**
```json
{
  "instruction": "Write a Python function to calculate the factorial of a number",
  "input": "5"
}
```

**Parameters:**
- `instruction` (required): Task description for code generation
  - Max length: 300 characters
  - Must not be empty
- `input` (optional): Additional input/context for the code
  - Default: `"< noinput >"`
  - Max length: 500 characters

**Response:**
```json
{
  "generated_code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))"
}
```

**Error Response:**
```json
{
  "detail": "Model inference failed"
}
```

**Example Usage (cURL):**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Write a Python function to calculate the factorial of a number",
    "input": "5"
  }'
```

**Example Usage (Python):**
```python
import requests

url = "http://localhost:8000/generate"
payload = {
    "instruction": "Write a Python function to calculate the factorial of a number",
    "input": "5"
}

response = requests.post(url, json=payload)
result = response.json()
print(result["generated_code"])
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

## � Docker & Cloud Deployment (Coming Soon)

### Containerization with Docker

Build a Docker image for consistent deployment:

```bash
# Build the Docker image
docker build -t mamba-code-gen:latest .

# Run the container locally
docker run -p 8000:8000 mamba-code-gen:latest
```

**Dockerfile** (to be created):
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment Options

**AWS Deployment:**
- **AWS EC2**: Deploy containerized API on EC2 instances
- **AWS ECS**: Use Elastic Container Service for orchestration
- **AWS SageMaker**: Alternative for managed inference
- **AWS Lambda**: Serverless inference (if model size permits)

**Google Cloud Deployment:**
- **Cloud Run**: Serverless container deployment
- **Compute Engine**: VM-based deployment
- **Vertex AI**: Managed ML model deployment

**Azure Deployment:**
- **Azure Container Instances (ACI)**: Quick container deployment
- **Azure App Service**: Web app hosting
- **Azure ML**: Managed ML inference

**Key Considerations:**
- Model checkpoint size (~50MB) - requires model artifact storage
- GPU availability - select instance types with GPU support
- Scaling - load balancing for high-traffic scenarios
- Cost optimization - use spot instances or auto-scaling groups

### Environment Variables for Deployment

```bash
MODEL_PATH=/app/artifacts/mamba_checkpoint_epoch10.pt
DEVICE=cuda  # or cpu
BATCH_SIZE=1
MAX_TOKENS=100
```

### Pre-Deployment Checklist

- [ ] Optimize model inference speed
- [ ] Add caching for repeated requests
- [ ] Implement rate limiting
- [ ] Add comprehensive logging/monitoring
- [ ] Set up health check endpoints
- [ ] Create `.dockerignore` to exclude unnecessary files
- [ ] Test API response times under load

## 🔮 Future Improvements

- [ ] Implement beam search for better code generation quality
- [ ] Add temperature and top-k sampling strategies
- [ ] ✅ Deploy REST API for inference (in progress)
- [ ] Dockerize the application
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Model quantization for edge deployment
- [ ] Extended context length (> 512 tokens)
- [ ] Caching layer for frequent requests
- [ ] Rate limiting and authentication

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
