import math
import os
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class TrainingDataArguments:
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    line_by_line: bool = field(default=False)
    block_size: int = field(default=-1)
    overwrite_cache: bool = field(default=False)


def prepare_dataset(args: TrainingDataArguments, tokenizer: GPT2Tokenizer, for_evaluation=False):
    if for_evaluation:
        file_path = args.eval_file
    else:
        file_path = args.train_file
    return TextDataset(
        tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
    )


def main():
    arg_parser = HfArgumentParser((TrainingDataArguments, TrainingArguments))
    data_args, training_args = arg_parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt2_config)

    data_args.block_size = gpt2_tokenizer.model_max_length


    training_dataset = prepare_dataset(data_args, tokenizer= gpt2_tokenizer)
    if training_args.do_eval:
        eval_dataset = prepare_dataset(data_args, tokenizer = gpt2_tokenizer, for_evaluation = True)
    else:
        eval_dataset = None
    data_collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm = False)

    trainer = Trainer(
        model=gpt2_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero:
            gpt2_tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
