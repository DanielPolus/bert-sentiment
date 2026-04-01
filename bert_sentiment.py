from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import torch
import matplotlib.pyplot as plt

dataset = load_dataset("imdb")
print(dataset)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }


train_dataset = IMDBDataset(dataset['train'], tokenizer)
test_dataset  = IMDBDataset(dataset['test'],  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
model = model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")


model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=['negative', 'positive']))


models = ['GloVe\n+ LR', 'BERT\nfine-tuning', 'TF-IDF\n+ LR']
scores = [0.840, 0.88, 0.888]
colors = ['#3498db', '#e74c3c', '#2ecc71']

plt.figure(figsize=(8, 5))
plt.bar(models, scores, color=colors)
plt.title('F1-score comparison across approaches')
plt.ylabel('F1-score')
plt.ylim(0.75, 0.95)
plt.tight_layout()
plt.savefig('comparison.png')
plt.show()
