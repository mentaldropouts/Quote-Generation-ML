import csv
import json
import re

# Function to read and parse the CSV file
def parse_csv(file_path):
    quotes = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            quote_text = row['quote']
            author = row['author']
            categories = row['category'].split(', ')

            quote_data = {
                'quote': quote_text,
                'author': author,
                'categories': categories
            }

            quotes.append(quote_data)

    return quotes

# Function to process quotes
def process_quotes(quotes):
    processed_quotes = []

    for quote in quotes:
        processed_quote = {
            'quote': quote['quote'].lower(),  # Convert to lowercase
            'author': quote['author'],
            'categories': quote['categories']
        }

        # Replace contractions
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
            "mightn't": "might not"
            # Add more contractions as needed
        }

        for contraction, replacement in contractions.items():
            processed_quote['quote'] = re.sub(fr"\b{re.escape(contraction)}\b", replacement, processed_quote['quote'])

        # Remove extra whitespaces
        processed_quote['quote'] = ' '.join(processed_quote['quote'].split())

        # Insert whitespace between word and punctuation (including commas)
        processed_quote['quote'] = re.sub(r'(?<=[^\s0-9])(?=[.,;!?])', ' ', processed_quote['quote'])
        processed_quote['quote'] = re.sub(r'(?<=,)(?=[^\s])', ' ', processed_quote['quote'])

        processed_quotes.append(processed_quote)

    return processed_quotes

# Example usage
file_path = 'data/quotes.csv'
parsed_quotes = parse_csv(file_path)

# Process quotes
processed_quotes = process_quotes(parsed_quotes)

# # Display the processed quotes
# for quote in processed_quotes:
#     print(f"Quote: {quote['quote']}")
#     print(f"Author: {quote['author']}")
#     print(f"Categories: {', '.join(quote['categories'])}")
#     print()

# Save processed quotes as JSON
processed_json_file_path = 'data\quotes.json'
with open(processed_json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(processed_quotes, jsonfile, ensure_ascii=False, indent=4)