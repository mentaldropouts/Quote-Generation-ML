# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Load Python script
with open('/content/gdrive/MyDrive/Quote-Generation-ML/ZBgpt2test.py', 'r') as f:
    python_code = f.read()

# Load JSON data
with open('/content/gdrive/MyDrive/Quote-Generation-ML/data/quotes.json', 'r') as f:
    json_data = f.read()

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW
from tqdm import tqdm
from torch.nn.functional import softmax
import random
import math
import json

MODEL_NAME = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

SPECIAL_TOKENS_DICT = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<quote>', '<author>', '<category>'],
}

tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
model.resize_token_embeddings(len(tokenizer))

class QuoteDataset(Dataset):
    def __init__(self, filename, tokenizer, seq_length=128, num_quotes=None):
        self.seq_length = seq_length
        quote_tkn = tokenizer.additional_special_tokens_ids[0]
        author_tkn = tokenizer.additional_special_tokens_ids[1]
        category_tkn = tokenizer.additional_special_tokens_ids[2]
        pad_tkn = tokenizer.pad_token_id
        eos_tkn = tokenizer.eos_token_id

        self.examples = []
        with open(filename, encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
            
            # Use only the first 'num_quotes' quotes if specified
            if num_quotes is not None:
                data = data[:num_quotes]

            for entry in data:
                quote = [quote_tkn] + tokenizer.encode(entry["quote"], max_length=seq_length//2-1)
                author = [author_tkn] + tokenizer.encode(entry["author"], max_length=seq_length//4-2) + [eos_tkn]
                category = [category_tkn] + tokenizer.encode(' '.join(entry["categories"]), max_length=seq_length//4-1) + [eos_tkn]

                tokens = quote + author + category + [pad_tkn] * ( seq_length - len(quote) - len(author) - len(category) )
                segments = [quote_tkn] * len(quote) + [author_tkn] * ( len(quote) + len(author) ) + [category_tkn] * ( seq_length - len(quote) - len(author) )
                labels = [-100] * (len(quote)+1) + author[1:] + [-100] * ( len(quote) + len(author) ) + category[1:] + [-100] * ( seq_length - len(quote) - len(author) - len(category) )

                self.examples.append((tokens, segments, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
      tokens, segments, labels = self.examples[item]
      
      # Pad sequences to the same length
      max_len = max(len(tokens), len(segments), len(labels))
      tokens += [tokenizer.pad_token_id] * (max_len - len(tokens))
      segments += [tokenizer.pad_token_id] * (max_len - len(segments))
      labels += [-100] * (max_len - len(labels))

      encoded_tokens = tokenizer.encode(
          tokens, 
          max_length=self.seq_length, 
          truncation=True,
          padding='max_length',
          return_tensors='pt'
      )

      return encoded_tokens.squeeze(), torch.tensor(segments), torch.tensor(labels)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        inputs, segments, labels = batch
        inputs, segments, labels = inputs.to(device), segments.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=(inputs != tokenizer.pad_token_id), labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def fit(model, optimizer, train_loader, val_loader, epochs, device):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}")

    print("Training finished.")

import torch

def sample_sequence(model, length, context, segments_tokens, num_samples=1):
    input_ids = torch.tensor(context, dtype=torch.long).unsqueeze(0)
    segments_tensor = torch.tensor(segments_tokens, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            token_type_ids=segments_tensor,
            max_length=input_ids.shape[-1] + length,  # Adjust the max_length based on the desired total length
            num_beams=num_samples,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=model.config.pad_token_id,  # Set pad_token_id
            attention_mask=torch.ones_like(input_ids),  # Set attention mask
        )

    return generated[:, input_ids.shape[-1]:].clone()



# Build the dataset and display the dimensions of the 1st batch for verification:
# Build the dataset using only the first 30 quotes for training
quote_dataset = QuoteDataset('/content/gdrive/MyDrive/Quote-Generation-ML/data/quotes.json', tokenizer, num_quotes=30)
print(next(iter(quote_dataset))[0].shape)

# Create data indices for training and validation splits:# Create data indices for training and validation splits:
indices = list(range(len(quote_dataset)))
random.seed(42)
random.shuffle(indices)
split = math.floor(0.1 * len(quote_dataset))
train_indices, val_indices = indices[split:], indices[:split]

# Build the PyTorch data loaders with a custom collate function:
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

import torch.nn.functional as F

def collate_fn(batch):
    inputs, segments, labels = zip(*batch)

    # Pad sequences dynamically within the batch
    max_len = max(seq.size(0) for seq in inputs)
    
    inputs = torch.stack([F.pad(seq, pad=(0, max_len - seq.size(0)), value=tokenizer.pad_token_id) for seq in inputs])
    segments = torch.stack([F.pad(seq, pad=(0, max_len - seq.size(0)), value=tokenizer.pad_token_id) for seq in segments])
    labels = torch.stack([F.pad(seq, pad=(0, max_len - seq.size(0)), value=-100) for seq in labels])

    return inputs, segments, labels



# Build the PyTorch data loaders with a custom collate function:
train_loader = DataLoader(quote_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(quote_dataset, batch_size=64, sampler=val_sampler, collate_fn=collate_fn)

# Fine-tune GPT2 for two epochs:
device = torch.device('cuda')
model.to(device)
optimizer = AdamW(model.parameters())
fit(model, optimizer, train_loader, val_loader, epochs=2, device=device)


# Move the model back to the CPU for inference:
model.to(torch.device('cpu'))

# Generate 20 samples of max length 20
context = "You know you are in love when you cannot fall asleep because reality is finally better than your dreams."
context_tkn = tokenizer.additional_special_tokens_ids[0]
author_tkn = tokenizer.additional_special_tokens_ids[1]
category_tkn = tokenizer.additional_special_tokens_ids[2]

input_ids = [context_tkn] + tokenizer.encode(context) + [author_tkn]
segments = [category_tkn] * 64
segments[:len(input_ids)] = [context_tkn] * len(input_ids)

input_ids += [category_tkn]

generated = sample_sequence(model, length=20, context=input_ids, segments_tokens=segments, num_samples=20)

print('\n\n--- Generated Quotes ---\n')

for g in generated:
    quote = tokenizer.decode(g.squeeze().tolist())
    quote = quote.split('<author>')[0].split('<quote>')[1]
    print(quote)


# Save the trained model
model.save_pretrained('/content/gdrive/MyDrive/Quote-Generation-ML/saved_model')
tokenizer.save_pretrained('/content/gdrive/MyDrive/Quote-Generation-ML/saved_model')
