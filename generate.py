import argparse
import numpy as np
import torch
from transformers import GPT2LMHeadModel,GPT2Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


###############################################################################
# This function controls the legth of the quotes that are produced by the model
###############################################################################
def adjust_length_to_model(length, max_sequence_length):
    return (
        max_sequence_length if length < 0 and max_sequence_length > 0 else
        max_sequence_length if 0 < max_sequence_length < length else
        MAX_LENGTH if length < 0 else
        length
    )

###############################################################################
# This is the main driver function for generating the quotes from the model
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )
    parser.add_argument("--repetition_penalty", type=float, default=1.0,)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="")
    # parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument('--stop_token', type=str, default=None)
    parser.add_argument("--max_return_sequences", type=int, default=1)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #args.seed = 42
    #set_seed(args)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    # Initialize the model and tokenizer

    model_class, tokenizer_class = GPT2LMHeadModel,GPT2Tokenizer

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path).to(args.device)
    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)
    input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.max_return_sequences,
    )



    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        #stoppingPoint = text.find(args.stop_token)
        #text = text[:stoppingPoint]
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]


        split_sequence = total_sequence.split(". ")
        sentences = [split + '.' for split in split_sequence]
        combined = " ".join(sentences[:-1])
        if combined != "":
            print("=== GENERATED SEQUENCE  ===")
            print(combined+'\n')


if __name__ == "__main__":
    main()
