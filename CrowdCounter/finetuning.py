"""
Fine Tuning Module

For fine-tuning LLMs on Datasets.
"""

import os
import torch
import timeit
import argparse
import warnings
from datetime import datetime
from imp_tokens import huggingface_token
warnings.filterwarnings("ignore")


#os.environ['WANDB_DISABLED'] = 'true'
os.environ['TRANSFORMERS_CACHE'] = '../cache/'




from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)

from peft import LoraConfig
from peft import PrefixTuningConfig

from trl import SFTTrainer, SFTConfig
from imp_tokens import huggingface_token
from dataloader import load_combination_dataset
from prompts import (counterspeech_prompt,
                     counterspeech_prompt_llama_two,
                     counterspeech_prompt_llama_three, 
                     type_specific_generation_prompt, 
                     type_specific_generation_prompt_llama_two, 
                     type_specific_generation_prompt_llama_three)


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

 
def load_tokenizer(params):
    """
    To load tokenizer of the corresponding model.
    Currently supports: flan-t5-base, llama-2-7b, falcon-7b, dialogpt-medium
    """
    
    model_id = params['model_path']
    model_name = params['model_name']

    if model_name == "flan-t5-base":
#         model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if model_name == "llama-2-7b":
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  trust_remote_code=True,
                                                  token=huggingface_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    if model_name == "llama-3b":
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  trust_remote_code=True,
                                                  token=huggingface_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    
    if model_name == "falcon-7b":
#         model_id = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    
    if model_name == "dialogpt-medium":
#         model_id = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    return tokenizer


def load_model(model_name:str, quantization_config = None, params = None):
    """
    To load models.
    Currently supports: flan-t5-base, Llama-2-7b-chat-hf, falcon-7b, dialogpt-medium
    """
    
    model_id = params['model_path']
        
    if model_name == "flan-t5-base":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    if model_name == "dialogpt-medium":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    if model_name == "llama-2-7b":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=huggingface_token
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    
    if model_name == "llama-3b":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    
    if model_name == "falcon-7b":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model.config.use_cache = False
     
    return model


def preprocess(dataset, tokenizer, params: dict):
    """
    Adding prompt and reformatting data.
    """
    print("inside preprocess", params['model_name'])
    prompt_to_be_used = get_prompts(params)

    if params['model_name'] == 'flan-t5-base':
        cols_to_be_removed = list(dataset['train'].features)
        
        def reformat_and_tokenize(sample, padding="max_length"):
            if params['type_specific']:
                inputs = [prompt_to_be_used.format(
                    type=', '.join(sample['total_types'][idx]), hate_speech=item) for idx,item in enumerate(sample["hatespeech"])]
            else:
                inputs = [prompt_to_be_used.format(hate_speech=item) for item in sample["hatespeech"]]
            model_inputs = tokenizer(inputs, padding=padding, truncation=True)

            labels = tokenizer(text_target=sample["counterspeech"], padding=padding, truncation=True)

            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        return dataset.map(reformat_and_tokenize, batched=True, remove_columns=cols_to_be_removed)
    else:
        def reformat(sample):
            if params['type_specific']:
                return {'text': prompt_to_be_used.format(
                    type=', '.join(sample['total_types']), hate_speech=sample['hatespeech']) + sample['counterspeech']}
            else:
                return {'text': prompt_to_be_used.format(
                    hate_speech=sample['hatespeech']) + sample['counterspeech']}

        return dataset.map(reformat)
    

def get_trainer(model, tokenizer, dataset, params: dict):
    """
    Preparing a Trainer instance for finetuning models.
    """
    now = datetime.now()
    output_dir = 'Logs/' + params['model_name'] + "_finetuning_" + '_'.join(str(now).split())    
    
    if params['model_name'] == 'flan-t5-base':
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model = model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
            
        training_arguments = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            num_train_epochs = params['num_epochs'] if params['num_epochs'] else 5,
            per_device_train_batch_size = params['train_batch_size'] if params['train_batch_size'] else 8,
            per_device_eval_batch_size = params['val_batch_size'] if params['val_batch_size'] else 4,
            dataloader_num_workers = params['num_workers'] if params['num_workers'] else 8,
            gradient_accumulation_steps = 4,
            gradient_checkpointing = params['gradient_checkpointing'],
            optim = "paged_adamw_32bit",
            logging_strategy = "steps",
            logging_steps = 100,
            learning_rate = params['lr'] if params['lr'] else 2e-4,
            weight_decay = 0.001,
            fp16 = params['fp16'] if params['fp16'] else False,
            bf16 = False,
            max_grad_norm = 0.3,
            max_steps = -1,
            warmup_ratio = 0.03,
            group_by_length = True,
            lr_scheduler_type = "cosine",
            save_strategy = "epoch",
            evaluation_strategy = "epoch", 
            metric_for_best_model = 'eval_loss',
            load_best_model_at_end = True,
            greater_is_better = False,
            report_to="wandb"
        )
            
        trainer = Seq2SeqTrainer(
            model = model,
            args = training_arguments,
            data_collator = data_collator,
            train_dataset = dataset["train"],
            eval_dataset = dataset["val"],
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02)]
        )
        
        return trainer
    
    else:
        peft_config = None
        
        dict_model_parts= {
            'llama-3b':["q_proj", "v_proj"],
            'llama-2-7b': ["q_proj", "v_proj"],
            'falcon-7b': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        }
        if params['peft']:
            if params['peft_type']=='lora':
                peft_config = LoraConfig(
                        lora_alpha=params['lora_alpha'] if params['lora_alpha'] else 16,
                        lora_dropout=0.1,
                        r=params['lora_rank'] if params['lora_rank'] else 64,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=dict_model_parts[params['model_name']]
                    )
            elif params['peft_type']=='prefixtuning':
                peft_config=PrefixTuningConfig(
                    peft_type="PREFIX_TUNING",
                    task_type="CAUSAL_LM",
                    num_virtual_tokens=20
                )


        print("the peft config" , peft_config)
                
        training_arguments = SFTConfig(
            output_dir = output_dir,
            num_train_epochs = params['num_epochs'] if params['num_epochs'] else 5,
            per_device_train_batch_size = params['train_batch_size'] if params['train_batch_size'] else 8,
            per_device_eval_batch_size = params['val_batch_size'] if params['val_batch_size'] else 4,
            dataloader_num_workers = params['num_workers'] if params['num_workers'] else 8,
            gradient_accumulation_steps = 4,
            gradient_checkpointing = params['gradient_checkpointing'],
            optim = "paged_adamw_32bit",
            logging_strategy = "steps",
            logging_steps = 5,
            learning_rate = params['lr'] if params['lr'] else 2e-4,
            weight_decay = 0.001,
            fp16 = params['fp16'] if params['fp16'] else False,
            bf16 = False,
            max_grad_norm = 0.3,
            max_steps = -1,
            warmup_ratio = 0.03,
            group_by_length = True,
            lr_scheduler_type = "cosine",
            save_strategy = "epoch",
            evaluation_strategy = "epoch", 
            metric_for_best_model = 'eval_loss',
            load_best_model_at_end = True,
            greater_is_better = False,
            report_to = "wandb"
        )
        
        #SFTTrainer which prints loss

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
            max_seq_length=512,
            packing=False,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02)]
        )
        
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)
        
        return trainer


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Fine-tuning script')

    parser.add_argument('--train_data', nargs='+', type=str, help='Training data sources')
    parser.add_argument('--val_data', nargs='+', type=str, default=[], help='Validation data sources')
    parser.add_argument('--test_data', nargs='+', type=str, default=[], help='Testing data sources')
    
    parser.add_argument('--train_sizes', nargs='+', type=int, help='Training data sizes')
    parser.add_argument('--val_sizes', nargs='+', type=int, default=[], help='Validation data sizes')
    parser.add_argument('--test_sizes', nargs='+', type=int, default=[], help='Testing data sizes')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--model_path', type=str, help='Model path')
    parser.add_argument('--type_specific', action='store_true', help='Enable type-specific generation')
    parser.add_argument('--train_batch_size', type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, help='Val batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--q4bit', action='store_true', help='Quantization')
    parser.add_argument('--peft', action='store_true', help='The flag stores whether to use PEFT or not')
    parser.add_argument('--peft-type', type=str, help='PEFT type -- LORA or PREFIXTUNING', default='lora')
    parser.add_argument('--fp16', action='store_true', help='Half Precision')
    parser.add_argument('--grad_ckpt', action='store_true', help='Gradient checkpointing')
    parser.add_argument('--lora-alpha', type=int, help='Lora alpha')
    parser.add_argument('--lora-rank', type=int, help='Lora rank')


    args = parser.parse_args()
    
    params = dict()
    params['model_name'] = args.model_name
    params['model_path'] = args.model_path 
    params['type_specific'] = args.type_specific
    params['train_batch_size'] = args.train_batch_size
    params['val_batch_size'] = args.val_batch_size
    params['num_epochs'] = args.num_epochs
    params['lr'] = args.lr
    params['fp16'] = args.fp16
    params['q4bit'] = args.q4bit
    params['peft'] = args.peft
    params['peft_type'] = args.peft_type
    params['num_workers'] = args.num_workers
    params['gradient_checkpointing'] = args.grad_ckpt
    params['lora_alpha'] = args.lora_alpha
    params['lora_rank'] = args.lora_rank

    # Loading Dataset
    print(args)
    print("="*100)
    dataset = load_combination_dataset(train_datasets=args.train_data, val_datasets=args.val_data,
                                       test_datasets=args.test_data, train_sizes=args.train_sizes,
                                       val_sizes=args.val_sizes, test_sizes=args.test_sizes,
                                       random_seed=args.random_seed, type_specific=args.type_specific)
    

    tokenizer = load_tokenizer(params)
    
    print("\nTransforming Dataset")
    transformed_dataset = preprocess(dataset, tokenizer, params)
    
    print(transformed_dataset)

    if params['q4bit']:
        print("\nLoading Model in Quantized 4 bit")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        
        model = load_model(params['model_name'], bnb_config, params)
        
    else:
        model = load_model(params['model_name'], params=params)
          
    # Finetuning
    print("\nFine-tuning Begins")
    t_0 = timeit.default_timer()
    
    trainer = get_trainer(model, tokenizer, transformed_dataset, params)
    trainer.train()
    
    t_1 = timeit.default_timer() 
    elapsed_time = round(t_1 - t_0, 3)
    print("Fine-tuning Ends")
    print(f"Elapsed time: {elapsed_time}s\n")
    
    # Saving Finetuned Models considering the following parameters type_specific, train_data, train_sizes, lora rank, lora alpha
    model_id = "Finetuned_Models/Generation/"
    if params['type_specific']:
        model_id += 'Type_specific_'

    for i in range(len(args.train_data)):
        model_id += args.train_data[i] + '('+str(args.train_sizes[i])+')_'

    if args.model_path:
        model_id += args.model_path.split('/')[1]
    else:
        model_id += args.model_name

#     if(params['peft']):
#         model_id += '_peft_'+params['peft_type']

#         if params['lora_rank']:
#             model_id += '_lora_rank_'+str(params['lora_rank'])
#         if params['lora_alpha']:
#             model_id += '_lora_alpha_'+str(params['lora_alpha'])

    trainer.model.save_pretrained(model_id)
    tokenizer.save_pretrained(model_id)
    print("Finetuned Model saved at", model_id)

    # save the params in the same location
    with open(model_id + "/params.txt", "w") as f:
        f.write(str(params))