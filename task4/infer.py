#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter("ignore")


# In[2]:


from transformers import AutoModelForCausalLM, Qwen2TokenizerFast
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from datasets import Dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_path = "./Dmeta-embedding-zh"
def get_dataset() -> Dataset:
    train_dataset = Dataset.load_from_disk("./train_dataset")
    return train_dataset
def to_doc(row: dict) -> Document:
    return Document(
        row["question"], metadata=row
    )
def get_embedding() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=embedding_path)
def get_db() -> Chroma:
    db = Chroma.from_documents(
        [to_doc(row) for row in get_dataset()],
        get_embedding()
    )
    return db


# In[3]:


model_path = "./Qwen2-7B-Instruct"
adapter_path = "./output"
max_new_tokens = 2048
batch_size = 16
voters = 5


# In[4]:

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


# In[9]:


def dump_entries(entries: list[Entry], file_path: str):
    import json
    with open(file_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, cls=EntryEncoder, ensure_ascii=False) + "\n")

def build_prompt_no_header(x: dict) -> str:
    problem = x["problem"]
    question = x["question"]
    reasoning = x["reasoning"]
    answer = x["answer"]
    options = x["options"]
    full_text = f"""### 题目        
{problem}

### 问题
{question}
{options}

### 分析过程
{reasoning}

### 答案
答案是：{answer}"""
    return full_text

def build_prompt(x: dict, db: Chroma) -> str:
    head = r"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。每个问题都保证能通过一系列基于形式逻辑的推理（包括同一律，矛盾律，排中律的使用等）得到确定的答案。请逐步分析问题，写出思考过程，并在最后一行输出答案，最后一行的格式为"答案是：A"或"答案是：B"或"答案是：C"或"答案是：D"等等。如果你做对了这个题目，你会获得的一亿奖金。"""
    tops = db.similarity_search(
        x["question"],
        k=2,
    )
    first_top = tops[0]
    second_top = tops[1]
    full = f"""{head}
这是一个例子：
{build_prompt_no_header(first_top.metadata)}

这是另一个例子：
{build_prompt_no_header(second_top.metadata)}

现在，你需要解决这个问题：
### 题目
{x["problem"]}

### 问题
{x["question"]}
{x["options"]}

### 分析过程"""
    return full

def get_prompts_of_entry(entry: Entry, db: Chroma) -> list[str]:
    prompts = []
    for question in entry.questions:
        prompt = build_prompt({
            "problem": entry.problem,
            "question": question.question,
            "options": format_options(question.options)
        }, db)
        prompts.append(prompt)
    return prompts

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
def get_model_and_sampling() -> tuple[LLM, SamplingParams, LoRARequest]:
    llm = LLM(model_path, model_path, enable_lora=True, dtype=torch.bfloat16)
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        n=voters,
    )
    lora = LoRARequest(
        "default",
        1,
        adapter_path
    )
    return llm, sampling, lora

def get_answer_from_raw(raw) -> str:
    raw = raw.text
    raw = raw.split("---")[0]
    raw = raw.split("### 题目")[0]
    raw = raw.split("### 问题")[0]
    remove_chars = ["*", "-"]
    for char in remove_chars:
        raw = raw.replace(char, "")
    import re
    match = re.search(r"答案是：[A-Z]", raw)
    if match:
        answer = match.group()
        answer = answer.split("：")[-1]
    else:
        answer = None
    return answer

def log_file(log_content: str):
    with open("./a.txt", "a") as f:
        f.write(log_content + "\n")

def generate_answers_batch(prompts: list[str], model: LLM, sampling: SamplingParams, lora: LoRARequest) -> list[str]:
    answers_unchecked = model.generate(
        prompts,
        sampling,
        lora_request=lora
    )
    answers = []
    for one_output in answers_unchecked:
        answer = [get_answer_from_raw(r) for r in one_output.outputs]
        log_file(
            "=" * 20 + "\n" + \
            "Prompt: \n" + prompts[answers_unchecked.index(one_output)] + "\n" + \
            "=" * 20 + "\n" + \
            ("=" * 20 + "\n").join(
                [f"Output: \b{o.text}" for o in one_output.outputs]
            )
        )
        filtered_answers = [ans for ans in answer if ans is not None]
        if not filtered_answers:
            most_common_answer = None
        else:
            most_common_answer = max(set(filtered_answers), key=filtered_answers.count)
        answer = most_common_answer
        if answer is None:
            print("Failed to get answer, default to A.")
            answer = "A"
        answers.append(answer)
    return answers
        

def generate_answers_inplace(entries: list[Entry], prompts: list[list[str]], model: LLM, sampling: SamplingParams, lora: LoRARequest):
    next_entry_index = 0
    next_question_index = 0
    
    total = sum([len(entry.questions) for entry in entries])



    progress = tqdm(total=(total // batch_size + (1 if total % batch_size != 0 else 0)))
    
    while next_entry_index < len(entries):
        batch_entries = []
        batch_prompts = []
        start_index_of_first_entry_question = next_question_index
        start_index_of_first_entry = next_entry_index
        while len(batch_prompts) < batch_size and next_entry_index < len(entries):
            entry = entries[next_entry_index]
            prompt = prompts[next_entry_index][next_question_index]
            batch_entries.append(entry)
            batch_prompts.append(prompt)
            next_question_index += 1
            if next_question_index >= len(entry.questions):
                next_question_index = 0
                next_entry_index += 1
        answers = generate_answers_batch(batch_prompts, model, sampling, lora)
        
        if len(answers) != batch_size:
            print("Answers length not equal to prompt entries length.")
            print("This should not happen.")
        
        i = 0
        next_entry_to_set = start_index_of_first_entry
        next_question_to_set = start_index_of_first_entry_question
        while i < len(answers):
            entry = entries[next_entry_to_set]
            entry.questions[next_question_to_set].answer = answers[i]
            i += 1
            next_question_to_set += 1
            if next_question_to_set >= len(entry.questions):
                next_question_to_set = 0
                next_entry_to_set += 1
        progress.update()
    progress.close()
        

# In[10]:


def main():
    entries = parse_file("./round1_test_data.jsonl")
    print("building db...")
    db = get_db()
    print("db built")
    print("building prompts...")
    prompts = [get_prompts_of_entry(entry, db) for entry in entries]
    # unload the embedding
    del db
    print(prompts[0])
    generator, sampling, lora = get_model_and_sampling()
    generate_answers_inplace(entries, prompts, generator, sampling, lora)
    dump_entries(entries, "./upload.jsonl")


# In[11]:


if __name__ == "__main__":
    main()


# In[ ]:




