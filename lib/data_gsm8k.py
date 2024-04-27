
# Borrowing from https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

import json
import os
import re
import torch
import random 

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"(.*?)####\s*(-?[0-9.,]+)", re.DOTALL)  # Ensure re.DOTALL is used if multiline handling is necessary
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    # Remove placeholder text enclosed in <<...>>
    completion = re.sub(r"<<.*?>>", "", completion)
    
    # Search for the pattern capturing text before and after '####'
    match = ANS_RE.search(completion)
    if match:
        # Extract rationale and answer, removing unnecessary spaces and commas
        rationale = match.group(1).strip()
        answer = match.group(2).strip().replace(",", "")  # Remove commas from numbers, if any
        return rationale, answer
    return INVALID_ANS  # Return invalid if no pattern matches


def get_gsm8k(traindata, testdata, nsamples, seed, seqlen, tokenizer, example= True):
    random.seed(seed)
    trainloader = []
    example =  traindata[0]
    question = example['question']
    rationale, answer = extract_answer(example['answer'])
    example = question + '\nRationale:' + rationale + '\nAnswer:' + answer

    for _ in range(nsamples): # no need of while true here 
        i = random.randint(1, len(traindata) - 1)
        sample = traindata[i]  # Get the complete sample
        question_instruction = sample['question']
        if example is True:
            prepend = f'Example: + {example}\n + Question:'
            question_instruction = prepend + question_instruction
        
        rationale, answer = extract_answer(sample['answer'])
        # rationale_enc = tokenizer(rationale, return_tensors='pt')
        # answer_enc = tokenizer(str(answer), return_tensors='pt') 
        
        question_instruction += '\nlets think step by step to get the rationale and the answer:'
        question_instruction_enc = tokenizer(question_instruction, return_tensors='pt')
                
        i = question_instruction_enc.input_ids.shape[1]
        inp = question_instruction_enc.input_ids[:, 0:i]
        trainloader.append((inp, str(rationale), str(answer)))

    testloader = []
    for sample in testdata:
        question = sample['question']
        rationale, answer = extract_answer(sample['answer'])
        # rationale_en = tokenizer(rationale, return_tensors='pt')

        question_instruction = question + '\n lets think step by step to get the rationale and the answer:\n'
        if example is True:
            prepend = f'Example: + {example}\n + Question:'
            question_instruction = prepend + question_instruction
        
        question_instruction_enc = tokenizer(question, return_tensors='pt')
        i = question_instruction_enc.input_ids.shape[1]
        inp = question_instruction_enc.input_ids[:, 0:i]
        rationale_en = rationale_en.input_ids
        testloader.append((question_instruction_enc, str(rationale_en), str(answer)))

    return trainloader, testloader


# (question_enc, rationale_enc, answer_enc, question_tar, rationale_tar, answer_tar) = trainloader
