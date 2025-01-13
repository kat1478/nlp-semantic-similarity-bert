# BERT-Based Semantic Similarity Classifier

This repository contains the implementation of a **BERT-based model** for evaluating semantic similarity between question pairs. The project focuses on fine-tuning a pre-trained BERT model on the [Quora Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset). The trained model can classify whether two questions have a similar meaning or not.

---

## 📂 Project Structure

- **`notebook.ipynb`**: Jupyter Notebook containing the complete workflow, from data preparation to model training and evaluation.
- **`Output_bert/`**: Directory containing the processed datasets and trained model files.
  - `train_data.csv`, `val_data.csv`, `test_data.csv`: Cleaned and tokenized datasets.
  - `train_tokens.json`, `val_tokens.json`, `test_tokens.json`: Tokenized representations of datasets.
  - `bert-base-uncased-trained/`: Directory containing the fine-tuned BERT model and tokenizer.
- **`metrics_summary.txt`**: Text file summarizing the model's performance metrics.
- **`plots/`**: Directory with visualizations such as confusion matrix and ROC curve.

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended for GPU acceleration)
- Install required libraries:
  ```bash
  pip install transformers scikit-learn pandas matplotlib seaborn
