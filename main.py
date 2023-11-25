import json

# Function to load parsed quotes from JSON
def load_quotes_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
        quotes = json.load(jsonfile)
    return quotes

# Example usage to load quotes
loaded_quotes = load_quotes_from_json("data\quotes.json")

# # Display the loaded quotes
# for quote in loaded_quotes:
#     print(f"Quote: {quote['quote']}")
#     print(f"Author: {quote['author']}")
#     print(f"Categories: {', '.join(quote['categories'])}")
#     print()

quote = loaded_quotes[0]
print(f"Quote: {quote['quote']}")
print(f"Author: {quote['author']}")
print(f"Categories: {', '.join(quote['categories'])}")
print()
