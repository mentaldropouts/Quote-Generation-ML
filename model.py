import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
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
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="gpt2",
    )
    cache_dir: Optional[str] = field(
        default=None,
    )


@dataclass
class DataTrainingArguments:
    train_data_file: Optional[str] = field(
        default=None
    )
    eval_data_file: Optional[str] = field(
        default=None,
    )
    line_by_line: bool = field(
        default=False,
    )

    block_size: int = field(
        default=-1,
    )
    overwrite_cache: bool = field(
        default=False
    )


def get_dataset(args: DataTrainingArguments, tokenizer: GPT2Tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(
        tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    set_seed(training_args.seed)

    config = GPT2Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        if trainer.is_world_process_zero:
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
