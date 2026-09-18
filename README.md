# MiniGPT — Character-Level Transformer Language Model

A small GPT-style **character-level language model** implemented in PyTorch and trained from scratch on the TinyShakespeare dataset.

This project was built as a learning exercise to understand how a Transformer language model works internally rather than relying on a pretrained GPT model or a high-level language-model library.

> **Project status:** Educational / experimental  
> **Framework:** PyTorch  
> **Training environment:** Google Colab with NVIDIA Tesla T4 GPU  
> **Model size:** 8.58M parameters

---

## What This Project Does

The model learns to predict the **next character** given the characters that came before it.

For example, during training the model sees sequences such as:

```text
First Citizen:
Before we proceed any further, hear me speak.
```

and learns the probability distribution of the next character at each position.

After training, the model can generate new text one character at a time using autoregressive generation.

The model also supports:

- Prompt-based text generation
- Temperature-controlled sampling
- Top-k sampling
- Train/validation loss tracking
- Validation perplexity tracking
- Automatic CPU/GPU selection

---

## Problem Statement

Large language models are built around the idea of **next-token prediction**: given previous tokens, predict what token comes next.

The goal of this project was to implement a small version of that idea from the ground up and understand the main components involved:

1. Turning text into numerical tokens
2. Creating training sequences
3. Representing tokens with embeddings
4. Computing causal self-attention
5. Building Transformer blocks
6. Training with next-character cross-entropy loss
7. Generating text autoregressively

This project therefore focuses more on **understanding the mechanics of a Transformer language model** than on maximizing benchmark performance.

---

## Dataset

The project uses **TinyShakespeare**, a small collection of Shakespeare text commonly used for educational character-level language-model experiments.

The notebook downloads the dataset from:

```text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Dataset details

| Property | Value |
|---|---:|
| Raw text size | 1,115,394 characters |
| Vocabulary | 65 unique characters |
| Train split | 90% |
| Validation split | 10% |
| Tokenization | Character-level |

The dataset is split sequentially: the first 90% is used for training and the final 10% for validation.

---

## Features

### Character-level tokenizer

The project builds a simple vocabulary from the characters present in the dataset.

```text
character → integer index
integer index → character
```

This keeps tokenization simple and makes the underlying language-modeling process easier to inspect.

### Causal self-attention

The custom attention implementation uses a causal mask so that a position can only attend to the current token and tokens before it.

This prevents the model from using future characters while predicting the next character.

### Multi-head attention

The attention mechanism uses:

- 8 attention heads
- 64 dimensions per head
- Learned Q, K and V projections

### Transformer blocks

The model contains 4 Transformer blocks with:

- Layer Normalization
- Causal self-attention
- Residual connections
- Feed-forward network
- GELU activation
- Dropout

### Autoregressive generation

After training, the model predicts one character, appends it to the sequence, and repeats the process.

Generation supports:

- `temperature`
- `top_k`
- configurable number of generated characters

---

## Model Architecture

The model is a small GPT-style Transformer implemented directly with PyTorch.

### Architecture

```text
Input characters
      ↓
Character IDs
      ↓
Token Embedding + Positional Embedding
      ↓
4 × Transformer Block
      │
      ├── LayerNorm
      ├── Causal Multi-Head Self-Attention
      ├── Residual Connection
      ├── LayerNorm
      ├── Feed-Forward Network
      └── Residual Connection
      ↓
Final LayerNorm
      ↓
Language Model Head
      ↓
Character logits
      ↓
Next-character prediction
```

### Model configuration

| Hyperparameter | Value |
|---|---:|
| Embedding dimension (`D_MODEL`) | 512 |
| Attention heads (`N_HEADS`) | 8 |
| Transformer layers (`N_LAYERS`) | 4 |
| Feed-forward dimension (`D_FF`) | 1024 |
| Context length (`CONTEXT`) | 256 |
| Dropout | 0.1 |
| Parameters | 8,576,512 |

The model also ties the token-embedding weights to the output language-model head.

---

## Training Process

The model was trained with the following setup:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Maximum learning rate | `3e-4` |
| Adam betas | `(0.9, 0.95)` |
| Weight decay | `0.1` |
| Warmup steps | 300 |
| Total steps | 3,000 |
| Evaluation interval | Every 100 steps |
| Gradient clipping | `1.0` |
| Batch size | 64 |
| Training context | 256 characters |

The learning rate starts with a warmup and then follows a cosine-decay schedule.

Training was completed in approximately **24.5 minutes** on the Tesla T4 runtime used in Google Colab.

---

## Evaluation Metrics

This project evaluates the language model using:

### Cross-entropy loss

The training objective is next-character cross-entropy loss.

Lower loss means the model is assigning higher probability to the correct next character.

### Perplexity

Validation perplexity is calculated as:

```python
perplexity = exp(validation_loss)
```

Perplexity provides another way of describing how uncertain the model is when predicting the next character.

This project does **not** use classification accuracy, because the task is language modeling rather than ordinary classification.

---

## Results

At the end of the recorded training run:

| Metric | Result |
|---|---:|
| Final training loss | 1.1493 |
| Final validation loss | 1.4753 |
| Final validation perplexity | 4.37 |
| Training steps | 3,000 |

The recorded validation perplexity decreased from **76.49 at step 0** to **4.37 at step 3,000**.

The notebook also records the following checkpoints:

| Step | Train Loss | Val Loss | Val Perplexity |
|---:|---:|---:|---:|
| 0 | 4.3383 | 4.3372 | 76.49 |
| 500 | 1.8445 | 1.9697 | 7.17 |
| 1000 | 1.4513 | 1.6573 | 5.25 |
| 1500 | 1.3017 | 1.5501 | 4.71 |
| 2000 | 1.2276 | 1.4991 | 4.48 |
| 2500 | 1.1767 | 1.4860 | 4.42 |
| 3000 | 1.1493 | 1.4753 | 4.37 |

The notebook also compares against a simple 65-character uniform baseline, for which the corresponding perplexity would be 65. This is only an illustrative baseline and should not be treated as a benchmark against modern language models.

---

## Example Output

One recorded generation used the prompt:

```text
ROMEO:
```

and produced:

```text
ROMEO:
Nay, it strange for my masters; but I am can report
to Camillo with them with thee. Let them not sit?

APHERS:
Not in the revenges, my lord.

SICINIUS:
How!

Messenger:
The stands of pardon the larks.
```

The output demonstrates that the model learned patterns such as Shakespeare-style formatting, character names, punctuation, line breaks, and word/character statistics.

However, the generated text is **not consistently coherent Shakespearean dialogue**. This is expected for a small character-level model trained for a short experimental run.

---

## Generation Settings

The notebook tests different sampling temperatures.

### Temperature = 0.8

Used for the main `ROMEO:` example.

### Temperature = 1.3

Produces more varied and less conservative generations.

Recorded example:

```text
POLIXENES:
I do tell that,
And in time: but when Corioli were he loves reod?
Bid thee. But, then I did
I said 'tou, strike:' well; and forsweave?
```

### Temperature = 0.3

Produces more conservative and repetitive output.

Recorded example:

```text
Second Servingman:
I know the prince of the house of York
And the state of the seas of the people,
Which not seem the sea of the sing of the country,
```

---

## Repository Structure

The intended cleaned repository structure is:

```text
MiniGPT-TinyShakespeare/
│
├── README.md
├── requirements.txt
├── notebook.ipynb
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
│
├── data/
│   └── README.md
│
├── images/
│   ├── training_curve.png
│   └── generation_examples.png
│
└── models/
    └── README.md
```

Some files in this structure are added as the project is cleaned up; the original experiment itself was created in Google Colab.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MiniGPT-TinyShakespeare.git
cd MiniGPT-TinyShakespeare
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open:

```text
notebook.ipynb
```

The notebook downloads the TinyShakespeare dataset automatically.

A GPU is recommended for training, but the code automatically falls back to CPU when CUDA is unavailable.

### Google Colab

The original experiment was developed in Google Colab. You can also upload/open the notebook in Colab and execute the cells sequentially.

---

## Limitations

This project is intentionally small and educational.

### Character-level tokenization

The model predicts individual characters instead of words or subword tokens. This makes the implementation easy to understand, but generation is less efficient than modern subword-based language models.

### Small model

The network contains approximately **8.58 million parameters**, which is tiny compared with modern large language models.

### Small dataset

The model is trained only on the TinyShakespeare corpus, so it does not learn broad language knowledge.

### Limited training

The recorded experiment runs for 3,000 steps. More training and tuning could potentially improve the generated text, but that was not tested in the current notebook.

### No systematic benchmark suite

The project currently evaluates loss, perplexity, and qualitative text generation. It does not include a broad language-model benchmark.

### No saved checkpoint in the original notebook

The original experiment does not save the trained model weights to a checkpoint. A cleaned repository can add checkpoint saving/loading as a future improvement.

---

## Future Improvements

Possible next steps include:

- Add model checkpoint saving and loading
- Add reproducibility through explicit random seeds
- Refactor notebook code into reusable Python modules
- Add a command-line generation script
- Run longer training experiments
- Compare different model sizes and context lengths
- Experiment with different learning rates and batch sizes
- Add more systematic evaluation
- Experiment with subword tokenization
- Improve generation quality and coherence

These are proposed improvements; they are **not results already achieved by the current experiment**.

---

## Development & AI Assistance

This project was developed as a learning project using **iterative AI-assisted coding / vibe coding**.

My process was:

1. I first studied the basic concepts involved in machine learning and language models.
2. I spent significant time thinking through the project requirements and writing an initial detailed prompt.
3. I used an AI coding assistant to generate implementation code and then copied the generated code into Google Colab cells.
4. I inspected the behavior of the code myself and used my own reasoning to identify possible errors, missing steps, or better ways to structure the notebook.
5. I asked targeted questions about individual issues and used the resulting suggestions to modify and retest specific cells.
6. I also discovered implementation details during this process — for example, recognizing that installation commands and imports can be handled directly inside notebook cells in Colab rather than treating the runtime only like a separate terminal environment.

### My role

I was responsible for:

- Understanding the project goal and the underlying ML concepts
- Designing the project requirements
- Deciding what to ask the AI coding assistant to build
- Testing the generated implementation
- Identifying issues and asking targeted debugging questions
- Evaluating the outputs and deciding what to change

### AI assistance

AI tools were used extensively for:

- Generating implementation code
- Explaining code and ML concepts
- Debugging individual parts
- Suggesting implementation changes
- Iteratively refining the notebook

This repository therefore represents an **AI-assisted learning and implementation project**, rather than a claim that every line was manually written from scratch.

---

## What I Learned

This project gave me practical exposure to:

- Character-level tokenization
- Embeddings
- Positional embeddings
- Query / Key / Value projections
- Multi-head self-attention
- Causal masking
- Transformer blocks
- Residual connections
- Layer normalization
- GELU
- AdamW
- Learning-rate warmup and cosine decay
- Cross-entropy loss
- Perplexity
- Autoregressive text generation
- Temperature and top-k sampling
- GPU-based PyTorch training

---

## Disclaimer

This is an educational experiment, not a production language model.

The purpose of the repository is to document the implementation, training experiment, results, limitations, and learning process behind a small Transformer language model.
