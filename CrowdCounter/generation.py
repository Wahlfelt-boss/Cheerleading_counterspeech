"""
To generate examples in test data using saved models.
"""

import json
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
from dataloader import load_combination_dataset, load_pure_dataset
from prompts import (counterspeech_prompt,
                     counterspeech_prompt_llama_two,
                     counterspeech_prompt_llama_three, 
                     type_specific_generation_prompt, 
                     type_specific_generation_prompt_llama_two, 
                     type_specific_generation_prompt_llama_three)

from imp_tokens import huggingface_token

#get time stamp as a string
import time
import os

os.environ['TRANSFORMERS_CACHE'] = '../cache/'

def get_prompts(params):
    """
    load the appropriate prompt for the particular model
    """
    prompt=None
    if params['type_specific']:
        if 'llama-2' in params['model_path'].lower():
            prompt=type_specific_generation_prompt_llama_two
        elif 'llama-3' in params['model_path'].lower():
            prompt=type_specific_generation_prompt_llama_three
        else:
            prompt=type_specific_generation_prompt
    else:
        if 'llama-2' in params['model_path'].lower():
            prompt=counterspeech_prompt_llama_two
        elif 'llama-3' in params['model_path'].lower():
            prompt=counterspeech_prompt_llama_three
        else:
            prompt=counterspeech_prompt
    return prompt

 

def generate(model, tokenizer, dataset, params):
    """
    To generate samples from dataset using the model.
    """
    num_samples = 1
    device = params['device']
    batch_size = params['batch_size']
    type_specific = params['type_specific']
    max_new = params['max_new_tokens']
    max_input = params['max_input_tokens']
   
    all_types = ["contradiction", "empathy_affiliation", "humour", "questions", "shaming", "warning-of-consequences"]
    
    prompt_type = get_prompts(params)
    
    generation = {"samples": {}, 'params':params}
    
    
    for batch_start in tqdm(range(0, len(dataset['test']['hatespeech']), batch_size)):
        batch_end = min(batch_start + batch_size, len(dataset['test']['hatespeech']))
        batch = dataset['test']['hatespeech'][batch_start:batch_end]
        cbatch = dataset['test']['counterspeech'][batch_start:batch_end]

        if type_specific:
            tbatch = dataset['test']['total_types'][batch_start:batch_end]
        
        hate_sentences_preprocessed = batch
        
        if type_specific:
            tbatch = dataset['test']['total_types'][batch_start:batch_end]
            cntr = {t: [] for t in all_types}
            for current_type in all_types:
                inputs = tokenizer(
                    [prompt_type.format(
                        type=current_type, hate_speech=hate_sentence) for hate_sentence in hate_sentences_preprocessed],
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=max_input,
                )
                
                # get the lengths of the inputs without padding



                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model.generate(input_ids=inputs['input_ids'],
                                             attention_mask=inputs["attention_mask"],
                                             max_new_tokens=max_new,
                                             do_sample=True,
                                             top_p=params['top_p'])
                    
                if params['type']=='seq2seq_lm':
                    replies=tokenizer.batch_decode(outputs,skip_special_tokens=True)    
                else:
                    replies=tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                # print(replies)
                # for i in range(len(batch)):
                #     input_batch = inputs["input_ids"][i * num_samples: (i + 1) * num_samples]
                #     response_batch = outputs[i * num_samples: (i + 1) * num_samples]
                    
                #     replies = []
                #     for input, response in zip(input_batch, response_batch):
                #         if params['type']=='seq2seq_lm':
                #             replies.append(tokenizer.decode(response, skip_special_tokens=True))
                #         elif params['type']=='causal_lm':
                #             replies.append(tokenizer.decode(response[input.shape[0]:], skip_special_tokens=True))
                #         else:
                #             print("wrong type of model")
                cntr[current_type].extend(replies)

            for i, hate_sentence in enumerate(batch):
                generation['samples'][batch_start + i] = {
                    "hatespeech": hate_sentences_preprocessed[i],
                    "types": {t: cntr[t][i] for t in all_types},
                    "org_hate": hate_sentence,
                    "org_counter": cbatch[i],
                    "org_type": tbatch[i]
                }
            
            # print(generation['samples'][batch_start]["types"])
        
        else:
            cntr = []
            
            inputs = tokenizer(
                [prompt_type.format(
                    hate_speech=hate_sentence) for hate_sentence in hate_sentences_preprocessed],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=max_input
            )
            
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new,
                    do_sample=True, 
                    top_p=params['top_p']
                )

            # for i in range(len(batch)):
            #     input_batch = inputs["input_ids"][i * num_samples: (i + 1) * num_samples]
            #     response_batch = outputs[i * num_samples: (i + 1) * num_samples]
                    
            #     replies = []
            #     for input, response in zip(input_batch, response_batch):
            #         if params['type']=='seq2seq_lm':
            #             replies.append(tokenizer.decode(response, skip_special_tokens=True))
            #         elif params['type']=='causal_lm':
            #             replies.append(tokenizer.decode(response[input.shape[0]:], skip_special_tokens=True))
            #         else:
            #             print("wrong type of model")
            if params['type']=='seq2seq_lm':
                replies=tokenizer.batch_decode(outputs,skip_special_tokens=True)        
            else:
                replies=tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                
            cntr.extend(replies)

            for i, hate_sentence in enumerate(batch):
                generation['samples'][batch_start+i] = {
                    "hatespeech": hate_sentences_preprocessed[i],
                    "org_hate": hate_sentence,
                    "counterspeech_model": [cntr[i]],
                    "org_counter": cbatch[i]
                }
                
        if i % 50 == 0:
            print(generation['samples'][batch_start + i])

    return generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generation script')
    parser.add_argument('--model_path', type=str, help='Model path')
    parser.add_argument('--save_path', type=str, help='Save path')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device: CPU/GPU')
    parser.add_argument('--causal_lm', action='store_true', help='For causal language modeling')
    parser.add_argument('--seq2seq_lm', action='store_true', help='For seq2seq language modeling')
    parser.add_argument('--test_data', nargs='+', type=str, default=[], help='Testing data sources')
    parser.add_argument('--test_sizes', nargs='+', type=int, default=[], help='Testing data sizes')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--q4bit', action='store_true', help='Quantization')
    parser.add_argument('--peft', action='store_true', help='peft')
    parser.add_argument('--type_specific', action='store_true', help='Enable type-specific generation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Max new tokens')
    parser.add_argument('--max_input_tokens', type=int, default=256, help='Max input tokens')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p value')
    args = parser.parse_args()
    params = {}
    params['batch_size'] = args.batch_size
    params['type_specific'] = args.type_specific
    params['q4bit'] = args.q4bit
    params['device'] = args.device
    params['max_new_tokens'] = args.max_new_tokens
    params['max_input_tokens'] = args.max_input_tokens
    params['top_p'] = args.top_p
    params['model_path'] = args.model_path
    if args.causal_lm:
        params['type']='causal_lm'
    elif args.seq2seq_lm:
        params['type']='seq2seq_lm'

    # Loading Dataset
    dataset = load_combination_dataset(test_datasets=args.test_data, 
                                       test_sizes=args.test_sizes if args.test_sizes else [
                                           None for _ in range(len(args.test_data))],
                                       random_seed=args.random_seed, type_specific=args.type_specific)
    
    # Loading Model
    print("model name: ", args.model_path)


    print("Loading Tokenizer .")
    if "flan-t5" in args.model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)    
    if "llama-2-7b" in args.model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  trust_remote_code=True,
                                                  token=huggingface_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
    if "llama-3" in args.model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  trust_remote_code=True,
                                                  token=huggingface_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
        
    if "falcon-7b" in args.model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
    
    if "dialogpt" in args.model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    

    print('\nLoading Model')
    bnb_config = None
    if params['q4bit']:
        print("\nLoading Model in Quantized 4 bit")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    if args.seq2seq_lm:
        if args.q4bit:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_path, 
                return_dict=True,
                quantization_config=bnb_config)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_path, 
                return_dict=True,
                quantization_config=bnb_config).cuda()
    elif args.causal_lm:
        if args.q4bit:
            model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        return_dict=True,
                        quantization_config=bnb_config,
                        token=huggingface_token
                    )
            model.bfloat16()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        return_dict=True,
                        quantization_config=bnb_config,
                        token=huggingface_token
                    ).cuda()
            
        model.generation_config.pad_token_id = tokenizer.pad_token_id

                        
        # else:
        #     model = AutoModelForCausalLM.from_pretrained(
        #         args.model_path, quantization_config=bnb_config).to('cuda:0')
            
#     model.to(params['device'])'
    
    # Generation
    print("\nGeneration Begins")
    generation = generate(model, tokenizer, dataset, params)
    print("Generation Ends")
    
    # Saving Generation
    save_path = args.save_path
    if not args.save_path:
        save_path = "Generated_Samples/"
        if params['type_specific']:
            save_path += 'Type_specific_'

        save_path += '_'.join(args.model_path.split('/')[2].split('_')[:-1]) + '_on_'

        for i in range(len(args.test_data)):
            if args.test_sizes:
                save_path += args.test_data[i] + '('+str(args.test_sizes[i])+')_'
            else:
                save_path += args.test_data[i] + '_'
        save_path += args.model_path.split('_')[-1] +'_' + time.strftime("%Y%m%d-%H%M%S") +'_.json'
    
    
    
    
    with open(save_path, 'w') as outfile:
        # if args.causal_lm and not args.type_specific:
        #     for v in generation['samples'].values():
        #         print(v)
        #         org = v['counterspeech_model'][0]
        #         cs = v['counterspeech_model'][0]
        #         #if cs.find('Assistant:') != -1:
        #         if cs.find('is:') != -1:
        #             print("Total output",cs)
        #             cs = cs[cs.find('" is:')+6:]
        #             #cs=cs.split("Assistant:", 1)[-1].strip()
        #             print("Counterspeech stripped:",cs)
        #             v['counterspeech_model'] = [cs]

        #sort the generation according to the hatespeech
        generation['samples'] = dict(sorted(generation['samples'].items(), key=lambda item: item[1]['hatespeech']))
        json.dump(generation, outfile, indent=4)
    
    print("\nGeneration Saved")