from datasets import Dataset

# Example data
data = {
    'head': ['entity1', 'entity2', 'entity3'],
    'relation': ['rel1', 'rel2', 'rel3'],
    'tail': ['entityA', 'entityB', 'entityC']
}

# Convert to Hugging Face dataset
dataset = Dataset.from_dict(data)


from transformers import LlamaTokenizer, LlamaForSequenceClassification
import torch
from torch.utils.data import DataLoader

# Load LLaMA tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained( "meta-llama/Meta-Llama-3-8B", cache_dir="models/LLaMA-HF/tokenizer")
model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    cache_dir="models/LLaMA-HF/model", num_labels=len(dataset['tail']))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_function(examples):
    inputs = [f"{h} {r}" for h, r in zip(examples['head'], examples['relation'])]
    targets = examples['tail']
    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(targets, padding=True, truncation=True, return_tensors="pt").input_ids
    model_inputs["labels"] = labels.squeeze()  # Make sure labels are correctly shaped
    return model_inputs

# Preprocess the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create a DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Define loss function
loss_fn = CrossEntropyLoss()

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Move input tensors to the GPU if available
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def score_candidate_entities(head, relation, candidates):
    model.eval()
    inputs = [f"{head} {relation} {candidate}" for candidate in candidates]
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.detach().cpu().numpy()
    return scores

# Example scoring
head = "entity1"
relation = "rel1"
candidates = ["entityA", "entityB", "entityC"]
scores = score_candidate_entities(head, relation, candidates)
print(scores)







