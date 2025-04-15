
# LoRA-RoBERTa for AG News Classification

> This project applies Low-Rank Adaptation (LoRA) to RoBERTa for efficient text classification on the AG News dataset with fewer than 1 million trainable parameters.

This repository contains the implementation and project report for fine-tuning `roberta-base` using **Low-Rank Adaptation (LoRA)** on the AG News dataset. The objective of the project was to achieve competitive performance while keeping trainable parameters under **1 million** — demonstrating the power of parameter-efficient adaptation in NLP.

---

## 🚀 Project Summary

- **Model:** `roberta-base` with LoRA adapters (frozen backbone)
- **Fine-tuning Technique:** [LoRA](https://arxiv.org/abs/2106.09685) via Hugging Face [PEFT](https://github.com/huggingface/peft)
- **Dataset:** [AG News](https://huggingface.co/datasets/ag_news)
- **Trainable Parameters:** ~870,000
- **Validation Accuracy:** **94.2%**

---

## 🧱 Architecture and Approach

This project used:
- **LoRA Adapters** injected into: `query`, `key`, `value`
- **LoRA config:** `r=7`, `alpha=32`, `dropout=0.1`, `bias='none'`
- Only LoRA layers are trainable; all RoBERTa weights remain frozen.
- Training loop managed by Hugging Face `Trainer`.

---

## 🧪 Evaluation Results

| Metric               | Score     |
|----------------------|-----------|
| Accuracy             | 94.2%     |
| Precision            | 94.3%     |
| Recall               | 94.2%     |
| F1 Score             | 94.1%     |
| Trainable Parameters | ~870,000  |

✅ A confusion matrix is generated and saved as `confusion_matrix.png`.

---

## 📁 Directory Structure

```
├── dlproj.py                 # Main script: loading, training, evaluation
├── README.md                # This file
├── final_model/             # Saved model checkpoints
├── inference_output.csv     # Predictions on test set
├── confusion_matrix.png     # Evaluation visualization
├── test_unlabelled.pkl      # (Optional) Unlabeled test data (if available)
├── Lora Roberta Report.tex  # Final project report (LaTeX)
```

---

## 📦 Installation

```bash
pip install torch transformers datasets peft scikit-learn matplotlib seaborn
```

---

## 🧠 How to Use

### 1. Train the Model

```bash
python dlproj.py
```

### 2. Run Inference

If `test_unlabelled.pkl` is present:
- Predictions are saved to `inference_output.csv`

---

## 📌 Future Work

- Apply [QLoRA](https://arxiv.org/abs/2305.14314) for quantized fine-tuning
- Extend to multi-label or multilingual datasets
- Use teacher-student distillation to improve efficiency
- Explore LoRA merging for multi-task models

---

## 🧾 Citations

- Hu et al., (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685  
- Wolf et al., (2020). *Transformers: State-of-the-art NLP*. EMNLP  
- Hugging Face PEFT: https://github.com/huggingface/peft  
- AG News Dataset: https://huggingface.co/datasets/ag_news

---

## 🧑‍💻 Authors

Your Name(s)  
Department of Computer Science  
Your University  
📧 your.email@example.edu
