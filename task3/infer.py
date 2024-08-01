#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter("ignore")


# In[2]:


from transformers import pipeline
from transformers import TextGenerationPipeline
from transformers import AutoModelForCausalLM, Qwen2TokenizerFast
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm


# In[3]:


model_path = "./autodl-fs/Qwen2-7B-Instruct"
adapter_path = "./output_inst"
max_new_tokens = 2048


# In[4]:


def get_pipeline() -> TextGenerationPipeline:
    import torch
    generator: TextGenerationPipeline =  pipeline("text-generation", model=model_path, tokenizer=model_path, device="cuda", torch_dtype=torch.bfloat16)
    from peft.config import PeftConfig
    conf = PeftConfig.from_pretrained(adapter_path)
    generator.model.add_adapter(conf)
    return generator


# In[5]:


import json

class EntryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Entry):
            return obj.__dict__
        if isinstance(obj, QuestionItem):
            return obj.__dict__
        if isinstance(obj, str):
            return obj
        return super().default(obj)

@dataclass
class QuestionItem:
    question: str
    options: list[str]
    answer: str | None = field(default=None)

@dataclass
class Entry:
    problem: str = field(default="")
    questions: list[QuestionItem] = field(default_factory=list)
    id: str = field(default="")


# In[6]:


def parse_file(file_path: str) -> list[Entry]:
    with open(file_path, "r") as f:
        lines = f.readlines()
    entries = []
    import json
    for line in lines:
        entry = json.loads(line)
        questions = []
        for question in entry["questions"]:
            questions.append(QuestionItem(**question))
        entries.append(Entry(problem=entry["problem"], questions=questions, id=entry["id"]))
    return entries


# In[7]:


def format_options(options):
    return '\n'.join(
        [
            f'{chr(ord("A") + i)}: {option}'
            for i, option in enumerate(options)
        ]
    )


# In[8]:


def generate_answer_inplace(entry: Entry, generator: TextGenerationPipeline):
    for question in entry.questions:
        got_answer = False
        attempts = 2
        while not got_answer and attempts > 0:
            prompt_template = """你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。每个问题都保证能通过一系列基于形式逻辑的推理（包括同一律，矛盾律，排中律的使用等）得到确定的答案。请逐步分析问题，写出思考过程，并在最后一行输出答案，最后一行的格式为"答案是：A"或"答案是：B"或"答案是：C"或"答案是：D"等等。题目如下：
### 题目:
{problem}

### 问题:
{question}
{options}

### 分析过程
"""
            prompt = prompt_template.format(
                **{
                    "problem": entry.problem,
                    "question": question.question,
                    "options": format_options(question.options)
                }
            )
            voter = 3
            result = generator(prompt, max_new_tokens=max_new_tokens, truncation=True, num_return_sequences=voter, do_sample=True)
            # answer = result[0]["generated_text"].split("答案是：")[-1].strip()
            counter = {}
            with open("a.txt", "a") as f:
                f.write("\n" + prompt + "\n" + "=" * 10)
            for i in range(voter):
                raw_response = result[i]["generated_text"]
                raw_response = raw_response.removeprefix(prompt)
                raw_response = raw_response.split("---")[0]
                raw_response = raw_response.split("### 题目")[0]
                raw_response = raw_response.split("### 问题")[0]
                import re
                # print("=" * 10)
                # print(raw_response)
                # print("=" * 10)
                with open("a.txt", "a") as f:
                    f.write(f"\n voter {i}  {'=' * 10}\n{raw_response}")
                
                match = re.search(r"答案是：[A-Z]", raw_response)
                if match:
                    answer = match.group()
                    answer = answer.split("：")[-1]
                else:
                    answer = None
                counter[answer] = counter.get(answer, 0) + 1
            answer = max(counter, key=counter.get)
            with open("a.txt", "a") as f:
                f.write(str(counter) + "\n" + "=" * 10)
            if answer:
                question.answer = answer.split("：")[-1]
                got_answer = True
            else:
                print("Failed to find answer in response.")
                print("Setting answer to C.")
                question.answer = "C"
            attempts -= 1


# In[9]:


def dump_entries(entries: list[Entry], file_path: str):
    import json
    with open(file_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, cls=EntryEncoder, ensure_ascii=False) + "\n")


# In[10]:


def main():
    generator = get_pipeline()
    entries = parse_file("./round1_test_data.jsonl")
    for entry in tqdm(entries):
        generate_answer_inplace(entry, generator)
    dump_entries(entries, "./upload.jsonl")


# In[11]:


if __name__ == "__main__":
    main()


# In[ ]:




