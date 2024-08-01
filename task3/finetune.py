#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch
from transformers import TextGenerationPipeline
from tqdm import tqdm
import os
from dataclasses import dataclass, field
from datasets import DatasetDict
from langchain.chains.base import Chain
from datasets import Dataset
from trl.trainer import ConstantLengthDataset
from peft import LoraConfig
from trl.trainer import SFTTrainer, SFTConfig
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast


# In[2]:


os.environ["WANDB_PROJECT"] = "fine-tune-logic-inference"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
max_new_tokens = 1024
# model_path = "/data/gongbu/LLMCraft/models/Qwen2-7B-Instruct"
model_path = "./autodl-fs/Qwen2-7B-Instruct"
output_path = "./output_inst"
micro_batch = 1


# In[3]:


def get_prompt_chain() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([
    ("system", '你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。每个问题都保证能通过一系列基于形式逻辑的推理（包括同一律，矛盾律，排中律的使用等）得到确定的答案。请逐步分析问题，写出思考过程，并在最后一行输出答案，最后一行的格式为"答案是：A"或"答案是：B"或"答案是：C"或"答案是：D"等等。如果你做对了这个题目，你会获得的一亿奖金。题目如下：'),
    ("user", """### 题目:
{problem}

### 问题:
{question}
{options}""")
])
    return prompt


# In[4]:


def format_options(options):
    return '\n'.join(
        [
            f'{chr(ord("A") + i)}: {option}'
            for i, option in enumerate(options)
        ]
    )


# In[5]:


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


# In[6]:

# In[7]:

def get_model_and_tokenizer() -> tuple[Qwen2ForCausalLM, Qwen2TokenizerFast]:
    model = Qwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
    return model, tokenizer


# In[8]:

# In[9]:

# In[10]:


def get_peft_config() -> LoraConfig:
    from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    from peft.utils.peft_types import TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen2"]
    )


# In[13]:


def get_training_args() -> SFTConfig:
    return SFTConfig(
        report_to="wandb",
        per_device_train_batch_size=micro_batch,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=8,
        gradient_accumulation_steps=2,
        output_dir=output_path,
        bf16=True,
        remove_unused_columns=False,
        logging_steps=20
    )


# In[14]:


def finetune():
    from accelerate import Accelerator
    acc = Accelerator()
    from accelerate.accelerator import AcceleratorState
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = micro_batch
    model, tokenizer = get_model_and_tokenizer()
    train_dataset = Dataset.load_from_disk("./train_dataset")
    def encode(x):
        head = r"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。每个问题都保证能通过一系列基于形式逻辑的推理（包括同一律，矛盾律，排中律的使用等）得到确定的答案。请逐步分析问题，写出思考过程，并在最后一行输出答案，最后一行的格式为"答案是：A"或"答案是：B"或"答案是：C"或"答案是：D"等等。如果你做对了这个题目，你会获得的一亿奖金。题目如下："""
        problem = x["problem"]
        question = x["question"]
        reasoning = x["reasoning"]
        answer = x["answer"]
        options = x["options"]
        full_text = f"""{head}
### 题目        
{problem}

### 问题
{question}
{options}

### 分析过程
{reasoning}

### 答案
答案是：{answer}"""
        encoded = tokenizer(full_text + tokenizer.eos_token, max_length=max_new_tokens, truncation=True, pad_to_multiple_of=8)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded
    train_dataset = train_dataset.map(
        lambda x: encode(x),
        batched=False
    ).remove_columns(['problem', 'question', 'options', 'reasoning', 'answer'])
    model, train_dataset = acc.prepare(
        model, train_dataset
    )
    
    peft_config = get_peft_config()
    training_args = get_training_args()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_path)


# In[15]:


def main():
    finetune()


# In[16]:


if __name__ == "__main__":
    main()

