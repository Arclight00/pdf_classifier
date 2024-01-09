import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
train_data = pd.read_csv('path_to_train_data.csv')
test_data = pd.read_csv('path_to_test_data.csv')

# Preprocess data
train_data['label'] = train_data['Is lighting product?'].apply(lambda x: 1 if x == 'Yes' else 0)
test_data['label'] = test_data['Is lighting product?'].apply(lambda x: 1 if x == 'Yes' else 0)

# Define a dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def train(self, train_dataset, test_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()
        return trainer

    def evaluate(self, trainer, test_dataset):
        return trainer.evaluate(eval_dataset=test_dataset)

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

    @staticmethod
    def plot_precision_recall(y_true, y_scores):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')

    @staticmethod
    def plot_roc_curve(y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')

# Usage
classifier = BertClassifier()
train_dataset = TextDataset(train_data['URL'].tolist(), train_data['label'].tolist(), classifier.tokenizer)
test_dataset = TextDataset(test_data['URL'].tolist(), test_data['label'].tolist(), classifier.tokenizer)

trainer = classifier.train(train_dataset, test_dataset)
eval_results = classifier.evaluate(trainer, test_dataset)

predictions = trainer.predict(test_dataset).predictions
predictions = np.argmax(predictions, axis=-1)

# Visualize
Visualizer.plot_confusion_matrix(test_dataset.labels, predictions)
plt.show()

Visualizer.plot_precision_recall(test_dataset.labels, predictions)
plt.show()

Visualizer.plot_roc_curve(test_dataset.labels, predictions)
plt.show()
