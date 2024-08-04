#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import Qwen2ForSequenceClassification
from transformers import Qwen2TokenizerFast
import torch
from datasets import Dataset
from dataclasses import dataclass, field


# In[2]:


model_path = "./Qwen2-0.5B"
max_length = 512
output_dir = "./scorer_output"


# In[3]:


def get_model_and_tokenizer() -> tuple[Qwen2ForSequenceClassification, Qwen2TokenizerFast]:
    model = Qwen2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
    )
    tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# In[4]:


@dataclass
class QuestionItem:
    question: str
    options: list[str]
    reasoning: str | None = field(default=None)
    answer: str | None = field(default=None)

@dataclass
class Entry:
    problem: str = field(default="")
    questions: list[QuestionItem] = field(default_factory=list)


# In[5]:


def get_train_dataset() -> Dataset:
    import pickle
    entries = pickle.load(open("./entries.pkl", "rb"))
    entries_false = pickle.load(open("./entries_false.pkl", "rb"))
    entries_aug = pickle.load(open("./entries_aug.pkl", "rb"))
    entries_aug_false = pickle.load(open("./entries_aug_false.pkl", "rb"))
    dataset = []
    for entry in (entries + entries_aug):
        for question in entry.questions:
            label = 1
            dataset.append({
                "reasoning": question.reasoning,
                "labels": label
            })
    for entry in (entries_false + entries_aug_false):
        for question in entry.questions:
            label = 0
            dataset.append({
                "reasoning": question.reasoning,
                "labels": label
            })
    return Dataset.from_list(dataset).shuffle(seed=42)


# In[6]:


def encode_dataset(dataset: Dataset, tokenizer: Qwen2TokenizerFast) -> Dataset:
    def encode(examples):
        encoded = tokenizer(
            examples["reasoning"],
            pad_to_multiple_of=8,
            max_length=max_length,
            truncation=True
        )
        return encoded
    return dataset.map(encode, batched=True).remove_columns(["reasoning"])


# In[7]:


import os
os.environ["WANDB_PROJECT"] = "fine-tune-logic-inference-scorer"


# In[8]:


from trl.trainer import SFTTrainer, SFTConfig
from transformers import TrainingArguments

def get_config() -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=5,
        seed=42,
        optim="lomo",
        lr_scheduler_type="cosine",
        report_to="wandb",
    )


# In[9]:


def train():
    from accelerate import Accelerator
    acc = Accelerator()
    model, tokenizer = get_model_and_tokenizer()
    train_dataset = get_train_dataset()
    train_dataset = encode_dataset(train_dataset, tokenizer)
    model, tokenizer, train_dataset = acc.prepare(model, tokenizer, train_dataset)
    config = get_config()
    from transformers import DataCollatorWithPadding
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.save_model("model")
    model.save_pretrained("./scorer", safe_serialization=True)


# In[10]:


if __name__ == "__main__":
    train()

