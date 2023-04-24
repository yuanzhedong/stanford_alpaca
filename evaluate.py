import argparse

parser = argparse.ArgumentParser(description='A program to add two numbers')
parser.add_argument('--ckpt', type=str, help='the first number')
parser.add_argument('--dataset', type=str, help='the second number')
args = parser.parse_args()


MODEL_PATH=args.ckpt
TEST_PATH=args.dataset

import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)


import transformers
from accelerate import init_empty_weights,load_checkpoint_and_dispatch

alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

prompter = Prompter("alpaca")

from transformers import GenerationConfig
def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=32,
    stream_output=False,
    **kwargs,
):
  prompt = prompter.generate_prompt(instruction, input)
  # print("prompt: ", prompt)
  inputs = alpaca_tokenizer(prompt, return_tensors="pt")
  input_ids = inputs["input_ids"].to(device)
  generation_config = GenerationConfig(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    num_beams=num_beams,
    **kwargs,
  )

  with torch.no_grad():
    generation_output = alpaca_model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
  
  s = generation_output.sequences[0]
  output = alpaca_tokenizer.decode(s, skip_special_tokens=True)

  return str(prompter.get_response(output)) 


import csv

TEST_DATA = []

# reading data from a csv file 'Data.csv'
with open(TEST_PATH, newline='') as file:
    reader = csv.reader(file)
    # store the headers in a separate variable,
    # move the reader object to point on the next row
    headings = next(reader)
    print(headings)
    for row in reader:
      TEST_DATA.append(row[:6])
  
print(len(TEST_DATA),TEST_DATA[0])


questions = [
    'Is the poster of this post stressful?',
    'Is the poster stressful?',
    'Is the poster likely to be stressful?',
    'Determine if the poster of this post is stressful.'
]

prompts = [
    '',
    'This person wrote this paragraph on social media. ',
    'Consider this post on social media to answer the question: ',
    'As a psychologist, read the post on social media and answer the question: ',
    'If you are a psychologist, read the post on social media and answer the question: ',
    'This person wrote this paragraph on social media. If you are a psychologist, read the post on social media and answer the question: ',
    'This person wrote this paragraph on social media. If you are a psychologist and consider the mental well-being condition expressed in this post, read the post on social media and answer the question: '
]

TASK = '\nOnly return Yes or No'

def pre_processing(content):
  each_data = alpaca_tokenizer( content.strip('[\'').strip('\']').replace('\n','') )
  each_data = alpaca_tokenizer.decode(each_data['input_ids'][:2000], skip_special_tokens=True)
  return each_data

def prompting(content, INSTRUCTION):
  res = "Post: \""+ content + '\"\n' + INSTRUCTION
  return res


results = {}

label = ['no', 'yes']
for i_p, each_p in enumerate(prompts):
  for i_q, each_q in enumerate(questions):
    cnt, cnt_correct, TP, FN, FP, TN = 0, 0, 0, 0, 0, 0

    INSTRUCTION = each_p + each_q + TASK
    for i in TEST_DATA:
      gt = label[int(i[-1])]
      each_content = pre_processing(i[4])
      each_input = prompting(each_content, INSTRUCTION)

      each_output = evaluate(each_input).strip().lower()

      if gt == 'yes':
        if 'yes' in each_output:
          TP += 1
          cnt_correct += 1
        else:
          FN += 1
      if gt == 'no':
        if 'no' in each_output:
          TN += 1
          cnt_correct += 1
        else:
          FP += 1
      cnt += 1
    
    each_accuracy = cnt_correct/cnt
    each_precision = TP/(TP+FP)
    each_recall = TP/(TP+FN)
    each_f1 = (2*each_recall*each_precision)/(each_precision+each_recall)
    each_name = 'dreaddit_prompt' + str(i_p) + '_question' + str(i_q)
    results[each_name + '_accuracy'] = each_accuracy
    results[each_name + '_precision'] = each_precision
    results[each_name + '_recall'] = each_recall
    results[each_name + '_f1'] = each_f1
    print(each_name + '_accuracy', each_accuracy)
    print(each_name + '_precision', each_precision)
    print(each_name + '_recall', each_recall)
    print(each_name + '_f1', each_f1)
    print('--------')
