# CrowdCounter: A benchmark type-specific multi-target counterspeech dataset [Accepted at CoNLL 2024]
This repository contains the code used in creating the annotation data

#### For more details about our paper

**Authors** : Punyajoy Saha, Abhilash Datta, Abhik Jana and Animesh Mukherjee

**Arxiv Link** :[CrowdCounter: A benchmark type-specific multi-target counterspeech dataset]()

**Paper Link** : https://arxiv.org/abs/2410.01400

# Abstract

Counterspeech presents a viable alternative to banning or suspending users for hate speech while upholding freedom of expression. However, writing effective counterspeech is challenging for moderators/users. Hence, developing suggestion tools for writing counterspeech is the need of the hour. One critical challenge in developing such a tool is the lack of quality and diversity of the responses in the existing datasets. Hence, we introduce a new dataset - CrowdCounter containing 3,425 hate speech-counterspeech pairs spanning six different counterspeech types (empathy, humor, questioning, warning, shaming, contradiction), which is the first of its kind. The design of our annotation platform itself encourages annotators to write type-specific, non-redundant and high-quality counterspeech. We evaluate two frameworks for generating counterspeech responses - vanilla and type-controlled prompts - across four large language models. In terms of metrics, we evaluate the responses using relevance, diversity and quality. We observe that Flan-T5 is the best model in the vanilla framework across different models. Type-specific prompts enhance the relevance of the responses, although they might reduce the language quality. DialoGPT proves to be the best at following the instructions and generating the type-specific counterspeech accurately. 

Please cite our paper in any published work that uses any of these resources.

~~~bibtex
@misc{saha2024crowdcounterbenchmarktypespecificmultitarget,
      title={CrowdCounter: A benchmark type-specific multi-target counterspeech dataset}, 
      author={Punyajoy Saha and Abhilash Datta and Abhik Jana and Animesh Mukherjee},
      year={2024},
      eprint={2410.01400},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.01400}, 
}
~~~

# Setup

### First set these environment variables before running any script
```
    export TF_CPP_MIN_LOG_LEVEL=1
    export WANDB_DISABLED=true 
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
    export TOKENIZERS_PARALLELISM=false
```

### For Finetuning LLMs:

- Filename: `finetuning.py`

- Finetuning flags:

      -h, --help            show this help message and exit
      --train_data TRAIN_DATA [TRAIN_DATA ...]
                            Training data sources
      --val_data VAL_DATA [VAL_DATA ...]
                            Validation data sources
      --test_data TEST_DATA [TEST_DATA ...]
                            Testing data sources
      --train_sizes TRAIN_SIZES [TRAIN_SIZES ...]
                            Training data sizes
      --val_sizes VAL_SIZES [VAL_SIZES ...]
                            Validation data sizes
      --test_sizes TEST_SIZES [TEST_SIZES ...]
                            Testing data sizes
      --random_seed RANDOM_SEED
                            Random seed
      --model MODEL         Model name
      --model_path MODEL_PATH
                            Model path
      --type_specific       Enable type-specific generation
      --train_batch_size TRAIN_BATCH_SIZE
                            Training batch size
      --val_batch_size VAL_BATCH_SIZE
                            Val batch size
      --num_epochs NUM_EPOCHS
                            Number of training epochs
      --num_workers NUM_WORKERS
                            Number of dataloader workers
      --lr LR               learning rate
      --q4bit               Quantization
      --peft                Peft
      --fp16                Half Precision
      --grad_ckpt           Gradient checkpointing
  
- Finetuned Models are saved in `Finetuned_Models` folder.

- For example: 
  ```
    CUDA_VISIBLE_DEVICES=1,2 python3 finetuning.py --train_data Reddit CONAN_New --val_data CONAN_New --train_sizes 100 1900 --val_sizes 100 --model flan-t5-base --num_epochs 1 --train_batch_size 8
  ```
    
<hr>

### For Generation:

- Filename: `generation.py`

- Generation flags:
  
      -h, --help            show this help message and exit
      --model_path MODEL_PATH
                            Model path
      --save_path SAVE_PATH
                            Save path
      --device DEVICE       Device: CPU/GPU
      --causal_lm           For causal language modeling
      --seq2seq_lm          For seq2seq language modeling
      --test_data TEST_DATA [TEST_DATA ...]
                            Testing data sources
      --test_sizes TEST_SIZES [TEST_SIZES ...]
                            Testing data sizes
      --random_seed RANDOM_SEED
                            Random seed
      --q4bit               Quantization
      --peft                peft
      --type_specific       Enable type-specific generation
      --batch_size BATCH_SIZE
                            Batch size
   
- Generated Samples are saved in `Generated_Samples` folder.

- For example:
    ```
    CUDA_VISIBLE_DEVICES=1 python3 generation.py --test_data Gab --test_sizes 100 --model_path "Reddit(100)_CONAN_New(1900)_flan-t5-base" --seq2seq_lm
    ```
    ```
    CUDA_VISIBLE_DEVICES=7 python3 generation.py --test_data Gab --causal_lm --batch_size 12 --peft --model_path "CONAN_New(100)_Gab(2700)_llama-2-7b"
    ```
    
<hr>

### For Scoring: (3 GPUS REQUIRED)

- Filename: `scoring.py`

- Scoring flags:

      -h, --help            show this help message and exit
      --file_name file_name    Name of the file in Generated_Samples folder

- Metrics Scores are saved in `Results` folder.

- For example:
  ```
    CUDA_VISIBLE_DEVICES=1,2,3 python scoring.py --file_name "Reddit(100)_CONAN_New(1900)_on_Gab(100)_flan-t5-base.json"
  ```
  
<hr>
