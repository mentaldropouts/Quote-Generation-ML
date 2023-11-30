import csv
import json
import re
import random

from transformers import GPT2Tokenizer

# Replace 'distilgpt2' with the actual model name you're using for quotes
QUOTES_MODEL_NAME = 'distilgpt2' 

quotes_tokenizer = GPT2Tokenizer.from_pretrained(QUOTES_MODEL_NAME)


# * Function to read and parse the CSV file
def parse_csv(file_path):
    quotes = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            quote_text = row['quote']
            author = row['author']
            categories = row['category'].split(', ')

            # * dict entry to add to json
            quote_data = {
                'quote': quote_text,
                'author': author,
                'categories': categories
            }

            quotes.append(quote_data)

    return quotes

# * Function to process quotes
def process_quotes(quotes):
    processed_quotes = []

    for quote in quotes:
        processed_quote = {
            'quote': quote['quote'].lower(),  # * Convert to lowercase
            'author': quote['author'],
            'categories': quote['categories']
        }

        # * Replace contractions
        contractions = {
            "wasn't": "was not",
            "isn't": "is not",
            "aren't": "are not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "mightn't": "might not",
            "i'm": "i am",
            "you've": "you have",
            "there's": "there is",
            "you'll": "you will",
            "it's": "it is",
            "you're": "you are"
            # * Add more contractions as needed
        }

        for contraction, replacement in contractions.items():
            processed_quote['quote'] = re.sub(fr"\b{re.escape(contraction)}\b", replacement, processed_quote['quote'])

        # * Remove extra whitespaces
        processed_quote['quote'] = ' '.join(processed_quote['quote'].split())

        # * Insert whitespace between word and punctuation (including commas)
        processed_quote['quote'] = re.sub(r'(?<=[^\s0-9])(?=[.,;!?])', ' ', processed_quote['quote'])
        processed_quote['quote'] = re.sub(r'(?<=,)(?=[^\s])', ' ', processed_quote['quote'])

        processed_quotes.append(processed_quote)

    return processed_quotes

# Function to filter out quotes over 100 words
def filter_long_quotes(quotes, max_tokens, tokenizer):
    return [quote for quote in quotes if len(tokenizer.encode(quote['quote'])) <= max_tokens]



# Function to randomly select a specified number of quotes
def select_random_quotes(quotes, num_quotes):
    return random.sample(quotes, min(num_quotes, len(quotes)))

# Example usage
file_path = 'data/quotes.csv'
parsed_quotes = parse_csv(file_path)

# Process quotes
processed_quotes = process_quotes(parsed_quotes)

# Filter out quotes over 100 words
max_gpt2_tokens = 128  # Set this to your GPT-2 model's maximum sequence length
filtered_quotes = filter_long_quotes(processed_quotes, max_gpt2_tokens, quotes_tokenizer)

# Randomly select 50,000 quotes
selected_quotes = select_random_quotes(filtered_quotes, 50000)

# # Display the filtered quotes
# for quote in filtered_quotes:
#     print(f"Quote: {quote['quote']}")
#     print(f"Author: {quote['author']}")
#     print(f"Categories: {', '.join(quote['categories'])}")
#     print()

# Save filtered quotes as JSON
filtered_json_file_path = 'data\quotes.json'
with open(filtered_json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(filtered_quotes, jsonfile, ensure_ascii=False, indent=4)