
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


# ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
# INVALID_ANS = "[invalid]"
# def extract_answer(completion):
#     match = ANS_RE.search(completion)
#     if match:
#         match_str = match.group(1).strip()
#         match_str = match_str.replace(",", "")
#         return match_str
#     else:
#         return INVALID_ANS
# Adjusted regex to capture both the rationale and the answer after '####'
ANS_RE = re.compile(r"(.*?)####\s*(\-?[0-9\.\,]+)")

INVALID_ANS = "[invalid]"

def extract_answer(completion):
    # Search for the pattern capturing text before and after '####'
    completion = re.sub(r"<<.*?>>", "", completion)
    # Search for the pattern capturing text before and after '####'
    match = ANS_RE.search(completion)
    if match:
        # Extract rationale and answer, removing unnecessary spaces and commas
        rationale = match.group(1).strip()
        answer = match.group(2).strip().replace(",", "")  # Remove commas from numbers, if any
        return rationale, answer
    return INVALID_ANS  # Return invalid if no pattern matches


def get_gsm8k(traindata, testdata, nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    trainloader = []
    example =  traindata[0]
    question = example['question']
    rationale, answer = extract_answer(example['answer'])
    example = question + '\nRationale:' + rationale + '\nAnswer:' + answer

    for _ in range(nsamples):
        while True:
            i = random.randint(1, len(traindata) - 1)
            sample = traindata[i]  # Get the complete sample
            question_rationale = sample['question']
            rationale, answer = extract_answer(sample['answer'])
            question_rationale += '\nRationale:' + rationale
            rationale_enc = tokenizer(rationale, return_tensors='pt')
            question_rationale_enc = tokenizer(question_rationale, return_tensors='pt')
            if question_rationale_enc.input_ids.shape[1] > seqlen:
                print('skipping sample, too long to encode')
                break
            answer_enc = tokenizer(str(answer), return_tensors='pt') 
        # i = random.randint(0, question_rationale_enc.input_ids.shape[1] - seqlen - 1)
        i = question_rationale_enc.input_ids[1]-rationale_enc.input_ids.shape[1]+2
        total_len = question_rationale_enc.input_ids.shape[1]
        j = min(i + seqlen, total_len)
        inp = question_rationale_enc.input_ids[:, i:j]
        tar = inp.clone()
        # tar[:, :-1] = -100
        tar[:, :-1] = -(rationale_enc.input_ids.shape[1] +3)
        trainloader.append((inp, tar))

    testloader = []
    for sample in testdata:
        question = sample['question']
        rationale, answer = extract_answer(sample['answer'])
        
        question_rationale = question + '\nRationale: ' + rationale
        question_rationale_enc = tokenizer(question_rationale, return_tensors='pt')
        padded_question_rationale = tokenizer.pad(question_rationale_enc, max_length=seqlen, padding='max_length', truncation=True)
        
        answer_enc = tokenizer(str(answer), return_tensors='pt')
        
        testloader.append((padded_question_rationale.input_ids, answer_enc))

    
def get_c4(traindata, valdata, nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# (question_enc, rationale_enc, answer_enc, question_tar, rationale_tar, answer_tar) = trainloader
