"""
Dataset Loading Module

We can load Individual/Pure Datasets or a Combination of Datasets.
"""

import random
from datasets import load_dataset, Dataset, DatasetDict


def load_pure_dataset(train_dataset:str, val_dataset:str, test_dataset:str,
                      train_size:int = None, val_size:int = None, test_size:int = None,
                      random_seed:int = 0):
    """Training and Testing dataset will be pure."""
        
    Train_dataset = load_dataset('json', data_files=f'./Datasets/{train_dataset}/Train.json')
    Val_dataset = load_dataset('json', data_files=f'./Datasets/{val_dataset}/Val.json')
    Test_dataset = load_dataset('json', data_files=f'./Datasets/{test_dataset}/Test.json')
    
    random.seed(random_seed)
    
    train = Train_dataset['train'][random.sample(range(0, len(Train_dataset['train'])), 
                                                 train_size if train_size is not None 
                                                 else len(Train_dataset['train']))]
    val = Val_dataset['train'][random.sample(range(0, len(Val_dataset['train'])), 
                                             val_size if val_size is not None 
                                             else len(Val_dataset['train']))]
    test = Test_dataset['train'][random.sample(range(0, len(Test_dataset['train'])), 
                                               test_size if test_size is not None 
                                               else len(Test_dataset['train']))]
    
    
      
    print("==================================")
    
    
    dataset = DatasetDict({'train':Dataset.from_dict(train),
                           'val':Dataset.from_dict(val),
                           'test':Dataset.from_dict(test)})
    return dataset





def load_combination_dataset(train_datasets:list = [], val_datasets:list = [], test_datasets:list = [],
                             train_sizes:list = [], val_sizes:list = [], test_sizes:list = [],
                             type_specific:bool = False, random_seed:int = 0):
    """Training and Testing dataset can be a combination of two or more datasets."""
    
    train = None
    val = None
    test = None
    random.seed(random_seed)
    ds = {}
    
#     keys = ['hatespeech', 'counterspeech']
    
#     if type_specific:
    keys = ['hatespeech', 'counterspeech', 'required_types', 'total_types']
    
    for i in range(len(train_datasets)):

        #load dataset using random sample such the the selected samples are the same across different runs

        if train_datasets[i] == "CrowdCounter":
            train_dataset = load_dataset('json', data_files=f'./Datasets/{train_datasets[i]}/Train.json')
        else:
            train_dataset = load_dataset('json', data_files=f'./Datasets/{train_datasets[i]}/Train_tune.json')
        if train is None:
            random.seed(random_seed)
            train = train_dataset['train'][random.sample(range(0, len(train_dataset['train'])),
                                                         train_sizes[i] if train_sizes[i] is not None 
                                                         else len(train_dataset['train']))]
        else:
            random.seed(random_seed)
            for k,v in train_dataset['train'][random.sample(range(0, len(train_dataset['train'])),
                                                            train_sizes[i] if train_sizes[i] is not None 
                                                            else len(train_dataset['train']))].items():
                if k in keys:
                    train[k] += v
       
    if len(train_datasets)>0:
        ds['train'] = Dataset.from_dict(dict(filter(lambda item: item[0] in keys, train.items())))
        
    for i in range(len(val_datasets)):
        val_dataset = load_dataset('json', data_files=f'./Datasets/{val_datasets[i]}/Val.json')
        if val is None:
            random.seed(random_seed)
            val = val_dataset['train'][random.sample(range(0, len(val_dataset['train'])),
                                                     val_sizes[i] if val_sizes[i] is not None 
                                                     else len(val_dataset['train']))]
        else:
            random.seed(random_seed)
            for k,v in val_dataset['train'][random.sample(range(0, len(val_dataset['train'])),
                                                          val_sizes[i] if val_sizes[i] is not None 
                                                          else len(val_dataset['train']))].items():
                if k in keys:
                    val[k] += v
    
    if len(val_datasets)>0:
        ds['val'] = Dataset.from_dict(dict(filter(lambda item: item[0] in keys, val.items())))
    
    for i in range(len(test_datasets)):
        test_dataset = load_dataset('json', data_files=f'./Datasets/{test_datasets[i]}/Test.json')
        if test is None:
            random.seed(random_seed)
            test = test_dataset['train'][random.sample(range(0, len(test_dataset['train'])),
                                                       test_sizes[i] if test_sizes[i] is not None 
                                                       else len(test_dataset['train']))]
        else:
            random.seed(random_seed)
            for k,v in test_dataset['train'][random.sample(range(0, len(test_dataset['train'])),
                                                           test_sizes[i] if test_sizes[i] is not None 
                                                           else len(test_dataset['train']))].items():
                if k in keys:
                    test[k] += v

    if len(test_datasets)>0:
        ds['test'] = Dataset.from_dict(dict(filter(lambda item: item[0] in keys, test.items())))
    
    
    dataset = DatasetDict(ds)
    return dataset