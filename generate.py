import argparse
import numpy as np
import torch
from transformers import GPT2LMHeadModel,GPT2Tokenizer

# Function to change length of generated quote to either user input or max input allowed
def changeLength(length, max_sequence_length):
    return (
        max_sequence_length if 0 < max_sequence_length < length else
        100 if length < 0 else
        length
    )

def main():
    # Add arguments that are (optionally) needed to run generate.py and impact quality of generated quotes
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_Location",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument('--stop_token', type=str, default=None)
    parser.add_argument("--max_return_sequences", type=int, default=1)

    #Parse arguments
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pull the model and tokenzier that were created from model.py
    modelUsed, tokenizerUsed = GPT2LMHeadModel,GPT2Tokenizer
    OurTokenizer = tokenizerUsed.from_pretrained(args.model_Location)
    OurModel = modelUsed.from_pretrained(args.model_Location).to(args.device)

    # Determine what length of quotes should be by calling function.
    # Ask user for input to be basis of generation
    args.length = changeLength(args.length, max_sequence_length=OurModel.config.max_position_embeddings)
    userContext = input("Enter the beginning of your quote or a topic:  ")

    # Tokenizes user input using the model created from model.py
    encodedInput = OurTokenizer.encode(userContext, add_special_tokens=True, return_tensors="pt").to(args.device)

    # Run the actual generation of quotes(that will still be tokenized) using HuggingFace function with our custom parameters
    output = OurModel.generate(
        input_ids= encodedInput,
        max_length=args.length + len(encodedInput[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.max_return_sequences,
    )

    # Go through each generated quote
    for iter, generatedSequence in enumerate(output):
        generatedSequence = generatedSequence.tolist()

        # Decotenize text
        text = OurTokenizer.decode(generatedSequence, clean_up_tokenization_spaces=True)

        # Find where to stop generation if given when running the file and change output to up to that stoppig point
        stoppingPoints = text.split(args.stop_token, 1) if args.stop_token else [text]
        text = stoppingPoints[0]

        # Add the prompt at the beginning of the sequence.
        total_sequence = userContext + text[len(OurTokenizer.decode(encodedInput[0], clean_up_tokenization_spaces=True)) :]

        # Remove sentence fragments from generated quotes
        split_sequence = total_sequence.split(". ")
        sentences = [split + '.' for split in split_sequence]
        combined = " ".join(sentences[:-1])

        #Print generated quotes
        if combined != "":
            print("--- QUOTE ---")
            print(combined+'\n')


if __name__ == "__main__":
    main()
