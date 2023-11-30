from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import csv
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW
from tqdm import tqdm
from torch.nn.functional import softmax
import random
import math

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
    def __init__(self, filename, tokenizer, seq_length=128):
        self.seq_length = seq_length
        quote_tkn = tokenizer.additional_special_tokens_ids[0]
        author_tkn = tokenizer.additional_special_tokens_ids[1]
        category_tkn = tokenizer.additional_special_tokens_ids[2]
        pad_tkn = tokenizer.pad_token_id
        eos_tkn = tokenizer.eos_token_id

        self.examples = []
        with open(filename, encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
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



# Build the dataset and display the dimensions of the 1st batch for verification:
quote_dataset = QuoteDataset('data\quotes.json', tokenizer)
print(next(iter(quote_dataset))[0].shape)
# Create data indices for training and validation splits:
indices = list(range(len(quote_dataset)))
random.seed(42)
random.shuffle(indices)
split = math.floor(0.1 * len(quote_dataset))
train_indices, val_indices = indices[split:], indices[:split]

# Build the PyTorch data loaders:
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(quote_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(quote_dataset, batch_size=64, sampler=val_sampler)

# Fine-tune GPT2 for two epochs:
device = torch.device('cpu')
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
