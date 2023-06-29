########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import pandas as pd
import re
from tqdm import tqdm
import os, sys, types, json, math, time, random
import jsonlines

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
except:
    pass
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

SAVE_DIR = "/home/kk/results/raven"
os.makedirs(SAVE_DIR, exist_ok=True)

models_dir = '/home/kk/MODELS/raven'
model_names = [os.path.join(models_dir, i) for i in os.listdir(models_dir)]

def output_rows(xlsx_path: str, sh_name: str) -> list[dict]:
    """Wczytuje plik xlsx i zwraca listę słowników (jeden wiersz -> jeden słownik)"""
    xlsx_df = pd.read_excel(xlsx_path, sheet_name=sh_name)
    xlsx_rows = xlsx_df.to_json(orient='records')
    xlsx_rows = xlsx_rows.lstrip('[').rstrip(']').split('{')
    xlsx_rows = ['{' + i for i in xlsx_rows[1:]]
    xlsx_rows = [i.replace('null', 'None') for i in xlsx_rows]
    xlsx_rows = [eval(i.rstrip(',').replace("true", "'true'").replace("false", "'false'")) for i in xlsx_rows]
    return xlsx_rows

def produce_dd(phones_data: list[dict]) ->list[dict]:

    # Prepare dict dataset
    dataset_dicts = []
    for phd in phones_data:
        phd['nazwasrodka'] = phd['nazwasrodka'].replace('\/', '').replace('\\/', '').replace('/', '')
        tokens = phd['nazwasrodka'].split(' ')
        new_dict = {'nazwasrodka' : phd['nazwasrodka'],
                    'taxonomy' : ''}
        if phd['T0'] and phd['T0'] in tokens:
            new_dict['taxonomy'] += phd['T0']
            new_dict['taxonomy'] += ' '

        if phd['T1'] and phd['T1'] in tokens:
            new_dict['taxonomy'] += phd['T1']
            new_dict['taxonomy'] += ' '
        for i in [2, 3, 4]:
            if phd[f'T{i}'] and phd[f'T{i}'] in tokens:
                new_dict['taxonomy'] += phd[f'T{i}']
                new_dict['taxonomy'] += ' '
        new_dict['taxonomy'] = new_dict['taxonomy'].rstrip(' ')
        dataset_dicts.append(new_dict)
    return dataset_dicts




# training examples for hte prompt
training_prompts = []
with jsonlines.open('/home/kk/ChatRWKV-kk/training_data/smartphones_training_sample_fuzzy.jsonl') as reader:
    for obj in reader:
        training_prompts.append(obj)

training_examples = []
for i in training_prompts:
    training_examples.append({'name': i['prompt'].replace('Generate a short name for the product: ', ''),
                              'taxonomy': i['completion']})

# zero-shot dataset for test
smartphone_test_data = '/home/kk/ChatRWKV-kk/test_data/smartfony_test.xlsx'
test_smartphones_source = output_rows(smartphone_test_data, "Sheet1")
test_smartphones = produce_dd(test_smartphones_source)

for i in test_smartphones:
    i['name'] = i['nazwasrodka']

oppos = [i for i in training_examples if 'oppo' in i['name']]
huaweis = [i for i in training_examples if 'huawei' in i['name']]
samsungs = [i for i in training_examples if 'samsung' in i['name']]
realmes = [i for i in training_examples if 'realme' in i['name']]
apples = [i for i in training_examples if 'apple' in i['name']]

random.shuffle(oppos)
random.shuffle(huaweis)
random.shuffle(samsungs)
random.shuffle(realmes)
random.shuffle(apples)

for MODEL_NAME in model_names:
    print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')


    print(f'Loading model - {MODEL_NAME}')
    model = RWKV(model=MODEL_NAME, strategy='cuda fp32') # !!! currenly World models will overflow in fp16 !!!
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # !!! update rwkv pip package to 0.7.4+ !!!

    ########################################################################################################



    results = []

    for ts_i in tqdm(range(len(test_smartphones))):
        ts = test_smartphones[ts_i]
        # Custom prompt - similar to what I've been using in other GPT-related experiments
        prompt = f"Here are a few examples of how one can generate a shorter version of a product name:\n" \
                 f"For product name: {huaweis[0]['name']}, the shorter version is: {huaweis[0]['taxonomy']}\n" \
                 f"For product name: {realmes[0]['name']}, the shorter version is: {realmes[0]['taxonomy']}\n"\
                 f"For product name: {samsungs[0]['name']}, the shorter version is: {samsungs[0]['taxonomy']}\n" \
                 f"For product name: {apples[0]['name']}, the shorter versionis: {apples[0]['taxonomy']}\n" \
                 f"Give me a shorter version for this product name:\n"
        input = ts['name']


        q= {'Instruction' : prompt, 'Input' : input}


        PAD_TOKENS = [] # [] or [0] or [187] -> probably useful


        out_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}
        state = None
        ctx = f'Instruction: {q["Instruction"].strip()}\n\nInput: {q["Input"].strip()}\n\nAnswer:' # !!! do not use Q/A (corrupted by a dataset) or Bob/Alice (not used in training) !!!
        print(ctx, end = '')
        output = {}
        for i in range(200):
            tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]

            out, state = pipeline.model.forward(tokens, state)
            for n in occurrence:
                out[n] -= (0.4 + occurrence[n] * 0.4) # repetition penalty

            token = pipeline.sample_logits(out, temperature=1.0, top_p=0.1)
            if token == 0: break # exit when 'endoftext'

            out_tokens += [token]
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = pipeline.decode(out_tokens[out_last:])
            if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): # only print when the string is valid utf-8 and not end with \n
                #print(tmp, end = '', flush = True)
                out_str += tmp
                out_last = i + 1

            if '\n\n' in tmp: # exit when '\n\n'
                out_str += tmp
                out_str = out_str.strip()
                break
        output['nazwasrodka'] = input
        output['taxonomy'] = ts['taxonomy']
        output['model_output'] = out_str
        results.append(output)

    with open(os.path.join(SAVE_DIR, os.path.basename(MODEL_NAME) + '.txt'), 'w') as jsntxt:
        json.dumps(jsntxt, results, indent=3)
