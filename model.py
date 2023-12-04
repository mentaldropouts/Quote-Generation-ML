import math
import os
from dataclasses import dataclass, field

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Create arguments that are needed to run this .py file
@dataclass
class TrainingDataArguments:
    train_file:  str = field(default=None)
    eval_file: str = field(default=None)
    line_by_line: bool = field(default=False)
    block_size: int = field(default=-1)
    overwrite_cache: bool = field(default=False)

# Process the text file to prepare it for tokenization
def prepare_dataset(args: TrainingDataArguments, tokenizer: GPT2Tokenizer, for_evaluation=False):
    if for_evaluation:
        file_path = args.eval_file
    else:
        file_path = args.train_file
    return TextDataset(
        tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
    )


def main():
    # Use built in library to process the arguments passed in command line
    arg_parser = HfArgumentParser((TrainingDataArguments, TrainingArguments))
    dataArgs, trainingArgs = arg_parser.parse_args_into_dataclasses()
    set_seed(trainingArgs.seed)

    # Read in pre-trained GPT2 model
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", config = GPT2Config.from_pretrained("gpt2"))

    # Set block size(input token count) to maximum possible
    dataArgs.block_size = gpt2_tokenizer.model_max_length

    # Call function.
    training_dataset = prepare_dataset(dataArgs, tokenizer= gpt2_tokenizer)
    if trainingArgs.do_eval:
        eval_dataset = prepare_dataset(dataArgs, tokenizer = gpt2_tokenizer, for_evaluation = True)
    else:
        eval_dataset = None

    #Using HuggingFave data collator to tokenize quotes, add padding, and prepare for training
    data_collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm = False)

    # Create training object using dataset inputs and gpt2 related frameworks
    trainer = Trainer(
        model=gpt2_model,
        args=trainingArgs,
        data_collator=data_collator,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
    )

    # If this was for training and save model(and not validation) save model 
    if trainingArgs.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero:
            gpt2_tokenizer.save_pretrained(trainingArgs.output_dir)


if __name__ == "__main__":
    main()
