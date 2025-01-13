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

# Fine-Tuning BERT for Semantic Similarity Detection

This project demonstrates how to fine-tune a pre-trained BERT model to classify question pairs as **Duplicate** or **Not Duplicate**, using the Quora Question Pairs dataset. The model identifies semantic similarities between question pairs, a key task in natural language processing (NLP).

---

## 📜 Dataset

**Source:** [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data)  
**Structure:**
- `question1`: Text of the first question.
- `question2`: Text of the second question.
- `is_duplicate`: Binary label (1 if similar, 0 otherwise).

---

## 🚀 Workflow

### 1. Data Preparation
- Cleaned raw data by removing special characters and handling missing values.
- Tokenized question pairs using **BERT Tokenizer** with a maximum sequence length of 128.

### 2. Model Fine-Tuning
- **Pre-trained model:** `bert-base-uncased`.
- Added a classification head with two output classes.
- Optimized using the **AdamW optimizer** with a learning rate of 2e-5 over 3 epochs.

### 3. Evaluation
- Evaluated on the test set using metrics:
  - **Accuracy**
  - **F1-Score**
  - **ROC-AUC**
- Generated **confusion matrix** and **ROC curve**.

---

## ⚙️ Usage

### Training the Model
1. **Clone the repository** and navigate to the directory:
   ```bash
   git clone https://github.com/your_username/bert-question-similarity.git
   cd bert-question-similarity
2. **Run the notebook**
   ```bash
   jupyter notebook notebook.ipynb
  Or, open in Google Colab for faster training.
### Save the Trained Model and Tokenizer
- After training the model, save it along with the tokenizer using the following commands:
  ```bash
  # Save the trained model
  model.save_pretrained("bert-base-uncased-trained")
  tokenizer.save_pretrained("bert-base-uncased-trained")
### Predicting with the Model
- To make predictions using the saved model and tokenizer:
  ```bash
  from transformers import BertForSequenceClassification, BertTokenizer

  # Load the model and tokenizer
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased-trained")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased-trained")

  # Example input
  inputs = tokenizer("What is AI?", "What is artificial intelligence?", return_tensors="pt", padding=True, truncation=True)
  outputs = model(**inputs)
  prediction = outputs.logits.argmax(dim=1).item()

  # Print the prediction
  print("Duplicate" if prediction == 1 else "Not Duplicate")

## 📊 Results
- **Validation Accuracy:** 90.79%
- **F1-Score:** Calculated using test data (details in `metrics_summary.txt`).
- **Confusion Matrix:** Available in `plots/confusion_matrix.png`.
- **ROC Curve:** Available in `plots/roc_curve.png`.

## 🗂 Saved Outputs
- **Trained Model**: Located in `bert-base-uncased-trained/`.
- **Processed Data:** CSV and tokenized JSON files in `Output_bert/`.
- **Visualizations:** Confusion matrix and ROC curve saved in the `plots/ directory`.
- **Metrics Summary:** A concise summary of model performance in `metrics_summary.txt`.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Acknowledgments
- Hugging Face for the transformers library.
- Quora for providing the dataset.
- Google Colab for offering free GPU resources.
Feel free to reach out with any questions or suggestions! 😊



