from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import torch
from tqdm import tqdm
import re
import pandas as pd
# from detoxify import Detoxify
import torch 
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
import difflib
import editdistance
import math
import spacy
from spacy.language import Language
import string
import torch
from nltk.tokenize import sent_tokenize
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertForMaskedLM
from transformers import glue_convert_examples_to_features, logging
from transformers.data.processors.utils import InputExample
from wmd import WMD
nltk.download('punkt')
from moverscore import get_idf_dict, word_mover_score
from typing import List, Union, Iterable
from itertools import zip_longest
import sacrebleu
from collections import defaultdict
import numpy as np


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons])


def preprocess_func(text):
    remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
    word_list=text_processor.pre_process_doc(text)
    word_list=list(filter(lambda a: a not in remove_words, word_list)) 
    sent=" ".join(word_list)
    sent = re.sub(r"[<\*>]", " ",sent)
    word_list=sent.split(" ")
    return word_list



def hate_refrences(data,test_set):          ###############returns pair of <hate,refrences>  
    hate  = []
    reply = []
    refrences = []
    for sample in data:
        ht , rep = sample[0] , sample[1]
        hate.append(ht)
        reply.append(rep)
    hate = list(set(hate))
    mp={}
    for ht_i in hate:
        refs = []
        for sample in data:
            ht_j , rep =  sample[0] , sample[1]
            if ht_j == ht_i:
                refs.append(rep)
        mp[ht_i] = refs
        refrences.append(refs)
    hate = list(set([x[0] for x in test_set]))
    refs = [mp[ht_i] for ht_i in hate]
    return hate,refs             # a given hate instance and refrences(replies) for metrics evaluation


# In[7]:


def training_corpus(train_set):    # returns training corpus
    replies = []
    for sample in train_set:
        rep = sample[1]
        replies.append(rep)
    replies = list(set(replies))
    return replies                # returns the sentences used while training 


from nltk import word_tokenize

def tokenize(sentence, max_sequence_length=None):
    token_sent = list(map(lambda x: str(x), list(word_tokenize(sentence))))
    if max_sequence_length is None:
        return token_sent
    else:
        return token_sent[:max_sequence_length]


def evaluate(params, model, test_dataloader, tokenizer, device):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
        inputs, labels = (batch[0], batch[0])
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
        
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return perplexity




def dummy(list_sent):
    return list_sent

###################################### BLEU_SCORE , METEOR #######################################
from nltk import word_tokenize

def nltk_metrics(generated_hypotheses, reference_sentences):
    """
    Calculate BLEU, GLEU, and METEOR scores using NLTK.

    Args:
    - generated_hypotheses (list): List of generated hypotheses.
    - reference_sentences (list): List of reference sentences for each generated hypothesis.

    Returns:
    - tuple: BLEU, GLEU, and METEOR scores.
    """
    hypotheses_tokens = [word_tokenize(h) for h in generated_hypotheses]
    references_tokens = [[word_tokenize(r) for r in refs] for refs in reference_sentences]

    total_hypotheses = len(hypotheses_tokens)
    bleu = gleu = meteor = 0.0

    for index, hypothesis_tokens in tqdm(enumerate(hypotheses_tokens), desc='NLTK Metrics Calculation:'):
        reference_tokens = references_tokens[index]

        bleu += sentence_bleu(reference_tokens, hypothesis_tokens, weights=(1.0, 1.0, 0, 0, 0.0),
                              smoothing_function=SmoothingFunction().method4)
        gleu += sentence_gleu(reference_tokens, hypothesis_tokens, min_len=1, max_len=2)
        meteor += meteor_score(reference_tokens, hypothesis_tokens)

    bleu /= total_hypotheses
    gleu /= total_hypotheses
    meteor /= total_hypotheses

    return bleu, gleu, meteor



############################################ JACCARD SIMILARITY #################################
def get_jaccard_sim(str1, str2):   
    if isinstance(str1, float) or isinstance(str2, float):
        return (-1)
    try:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print((str1))
        print(type(str2))
        return 0


############################################### NOVELTY #########################################
def get_novelty(sent,training_corpus):
    max_overlap = 0
    for instance in training_corpus:
        max_overlap = max(max_overlap,get_jaccard_sim(instance,sent))
    return 1-max_overlap

def avg_novelty(sentences,training_corpus):
    avg = 0
    for sent in tqdm(sentences,total=len(sentences),desc='Novelty:'):
        avg += get_novelty(sent,training_corpus)
    avg = (avg/float(len(sentences)))
    return avg



############################################### DIVERSITY ########################################
def get_diversity(sentences):
    avg = 0.0
    for i in tqdm(range(len(sentences)),desc='Diversity:'):
        max_overlap = 0
        for j in range(len(sentences)):
            if i!=j:
                max_overlap = max(max_overlap,get_jaccard_sim(sentences[i],sentences[j]))
        avg = avg + (1-max_overlap)
    avg = (avg/len(sentences))
    return avg
    
def diversity_and_novelty(training_corpus,gen_replies):
    diversity = get_diversity(gen_replies)
    novelty   = avg_novelty(gen_replies,training_corpus)
    return diversity,novelty





############################################## HEAVY METRICS ########################################
class Bleurt():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu, gpu='cuda:0'):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
   
    def score(self,params):
        hypo = params[0]  # a list of generated_hypothesis   
        refs = params[1]  # a list of refrences for particular_refrences    
        device = self.device
        list_ids=[]
        hypo_all=[]
        refs_all=[]
        for step in range(len(hypo)):
            list_ids.append(step)
            hypo_all.append(hypo[step])
            refs_all.append(refs[step])
        
        print("Collected all points")
        
        scores_all=[]
        for i in tqdm(range(0, len(hypo_all), self.batch_size)):
            with torch.no_grad():
    
                inputs = self.tokenizer(refs_all[i:i+self.batch_size], hypo_all[i:i+self.batch_size], return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)

                if(self.use_gpu):   
                    scores = self.model(input_ids=inputs['input_ids'].to(device),
                                           attention_mask=inputs['attention_mask'].to(device),
                                           token_type_ids=inputs['token_type_ids'].to(device))[0].squeeze().cpu().numpy()
                else:
                    scores = self.model(input_ids=inputs['input_ids'],
                                           attention_mask=inputs['attention_mask'],
                                           token_type_ids=inputs['token_type_ids'])[0].squeeze().cpu().numpy()
                
                if scores.ndim == 0:
                    scores = [scores.item()]  # Convert scalar to a list with a single element
                else:
                    scores = list(scores)
                scores_all+=scores
        
        
        
        df=pd.DataFrame(list(zip(list_ids, scores_all)), columns=['ids', 'scores'])
        df_mean=df.groupby(['ids']).mean()        
        
        print(df_mean.head(5))
        
        mean_bleurt_score = np.mean(list(df_mean['scores']))
        return mean_bleurt_score

    
#### Without REFERENCE
class Argument_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu, gpu='cuda:0'):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device=torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
    def scoring(self, hypo):
        device = self.device
        scores_all=[]
        print(hypo[0:5])
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            
            with torch.no_grad():
            
              inputs = self.tokenizer(hypo[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
              
              if(self.use_gpu):    
                  scores = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device))[0].squeeze()
              else:
                  scores = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])[0].squeeze()
                    
              
              scores = torch.softmax(scores.T, dim=0).T.cpu().numpy()
              
              try:
                scores_all+=list(scores[:,1])
              except:
                continue
        
#         with torch.no_grad():
#             scores = self.model(**self.tokenizer(hypo, return_tensors='pt',truncation=True, padding=True, max_length=64))[0].squeeze()
#             scores = torch.softmax(scores.T, dim=0).T.cpu().numpy()
        
        return np.mean(scores_all)
    

    

class Dialog_upvote_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu, gpu='cuda:0'):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device=torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
    def scoring(self,hypo,hate):
        device = self.device
        hypo_hate=[]
        
        print(hypo[0:5],hate[0:5])

        for i in range(len(hypo)):
            str1=hate[i]+'<|endoftext|>'+hypo[i]
            hypo_hate.append(str1)
        
        device = self.device

        scores_all=[]
        for i in tqdm(range(0, len(hypo_hate), self.batch_size)):
            with torch.no_grad():
                inputs = self.tokenizer(hypo_hate[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
                if(self.use_gpu):    
                      results = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device),return_dict=True)
                else:
                      results = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],return_dict=True)
                scores=list(torch.sigmoid(results.logits).cpu().numpy())
                scores_all+=scores
#         print(scores[0:5])
        return np.mean(scores_all)


class Counter_argument_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu, gpu='cuda:0'):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=cache_path)
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()
    
    def scoring(self,hypo,hate):
        print(hypo[0:5],hate[0:5])
        device = self.device
        scores_all=[]
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
                inputs = self.tokenizer(text=hate[i:i+self.batch_size],text_pair=hypo[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
                if(self.use_gpu):    
                      scores = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device),return_dict=True)
                else:
                      scores = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],return_dict=True)
                scores = torch.softmax(scores['logits'].T, dim=0).T.cpu().numpy()
                scores_all+=list(scores[:,1])
        
        print(scores_all[0:5])
        return np.mean(scores_all)


class Toxicity_model():
    def __init__(self,max_length, batch_size,use_gpu, gpu='cuda:0'):
        self.max_length= max_length
        self.batch_size = batch_size
        self.use_gpu=use_gpu
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            self.model = Detoxify('unbiased', device=self.device)
        else:
            self.model = Detoxify('unbiased', device='cpu')
        
    def scoring(self,hypo):
        scores_all=[]
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
                scores=self.model.predict(hypo[i:i+self.batch_size])
                scores_all+=list(scores['toxicity'])
        print(scores_all[0:5])
        return np.mean(scores_all)
    
    
###################################### Detox scores using HateXplain #################################################
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class Model_Rational_Label(BertPreTrainedModel):
     def __init__(self,config):
        super().__init__(config)
        self.num_labels=2
        self.impact_factor=0.8
        self.bert = BertModel(config,add_pooling_layer=False)
        self.bert_pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()        
#         self.embeddings = AutoModelForTokenClassification.from_pretrained(params['model_path'], cache_dir=params['cache_path'])
        
     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, attn=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        # out = outputs.last_hidden_state
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
#         mean_pooling = torch.mean(out, 1)
#         max_pooling, _ = torch.max(out, 1)
#         embed = torch.cat((mean_pooling, max_pooling), 1)
        embed=self.bert_pooler(outputs[0])
        y_pred = self.classifier(self.dropout(embed))
        loss_token = None
        loss_label = None
        loss_total = None
        
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                )
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total=self.impact_factor*loss_token
            
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if(loss_total is not None):
                loss_total+=loss_label
            else:
                loss_total=loss_label
        if(loss_total is not None):
            return y_pred, logits, loss_total
        else:
            return y_pred, logits

# class Args():
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args = Args()

class Toxic_HateXplain_scoring():
    def __init__(self, model_path,cache_path,max_length, batch_size,use_gpu, gpu='cuda:0'):
        self.max_length= max_length
        self.batch_size= batch_size
        self.use_gpu   = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",cache_dir=cache_path)
        self.model     =  Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            self.model.to(self.device) 
        self.model.eval()
    
    def scoring(self,hypo,hate):
        scores_all=[]
        device = self.device
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():
                inputs = self.tokenizer(text=hypo[i:i+self.batch_size],return_tensors='pt',truncation=True, padding=True, max_length=self.max_length)
                if(self.use_gpu):    
                      logits, _ = self.model(input_ids=inputs['input_ids'].to(device),attention_mask=inputs['attention_mask'].to(device))
                else:
                      logits, _ = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                scores = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
                scores_all+=list(scores[:,1])
        
        print(scores_all,len(scores_all))
        return np.mean(scores_all)
    
    
##############################################################################################################################
##                                          GRUEN SCORE                                                                     ##
##############################################################################################################################

class Gruen:
    def __init__(self, use_gpu, gpu='cuda:0'):
        self.use_gpu   = use_gpu
        self.device = torch.device("cpu")
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            
    def preprocess_candidates(self, candidates):
        for i in range(len(candidates)):
            candidates[i] = candidates[i].strip()
            candidates[i] = '. '.join(candidates[i].split('\n\n'))
            candidates[i] = '. '.join(candidates[i].split('\n'))
            candidates[i] = '.'.join(candidates[i].split('..'))
            candidates[i] = '. '.join(candidates[i].split('.'))
            candidates[i] = '. '.join(candidates[i].split('. . '))
            candidates[i] = '. '.join(candidates[i].split('.  . '))
            while len(candidates[i].split('  ')) > 1:
                candidates[i] = ' '.join(candidates[i].split('  '))
            myre = re.search(r'(\d+)\. (\d+)', candidates[i])
            while myre:
                candidates[i] = 'UNK'.join(candidates[i].split(myre.group()))
                myre = re.search(r'(\d+)\. (\d+)', candidates[i])
            candidates[i] = candidates[i].strip()
        processed_candidates = []
        for candidate_i in candidates:
            sentences = sent_tokenize(candidate_i)
            out_i = []
            for sentence_i in sentences:
                if len(
                        sentence_i.translate(
                            str.maketrans('', '', string.punctuation)).split()
                ) > 1:  # More than one word.
                    out_i.append(sentence_i)
            processed_candidates.append(out_i)
        return processed_candidates


    """ Scores Calculation """


    def get_lm_score(self, sentences):
        device = self.device
        def score_sentence(sentence, tokenizer, model):
            # if len(sentence.strip().split()) <= 1:
            #     return 10000
            tokenize_input = tokenizer.tokenize(sentence)
            if len(tokenize_input) > 510:
                tokenize_input = tokenize_input[:510]
            input_ids = torch.tensor(
                tokenizer.encode(tokenize_input)).unsqueeze(0).to(device)
            with torch.no_grad():
                loss = model(input_ids, labels=input_ids)[0]
            return math.exp(loss.item())

        model_name = 'bert-base-cased'
        model = BertForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(model_name)
        lm_score = []
        for sentence in tqdm(sentences):
            if len(sentence) == 0:
                lm_score.append(0.0)
                continue
            score_i = 0.0
            for x in sentence:
                score_i += score_sentence(x, tokenizer, model)
            score_i /= len(sentence)
            lm_score.append(score_i)
        return lm_score


    def get_cola_score(self, sentences):
        device = self.device
        def load_pretrained_cola_model(model_name,
                                       saved_pretrained_CoLA_model_dir):
            config_class, model_class, tokenizer_class = (
                BertConfig, BertForSequenceClassification, BertTokenizer)
            config = config_class.from_pretrained(saved_pretrained_CoLA_model_dir,
                                                  num_labels=2,
                                                  finetuning_task='CoLA')
            tokenizer = tokenizer_class.from_pretrained(
                saved_pretrained_CoLA_model_dir, do_lower_case=0)
            model = model_class.from_pretrained(
                saved_pretrained_CoLA_model_dir,
                from_tf=bool('.ckpt' in model_name),
                config=config).to(device)
            model.eval()
            return tokenizer, model

        def evaluate_cola(model, candidates, tokenizer, model_name):

            def load_and_cache_examples(candidates, tokenizer):
                max_length = 128
                examples = [
                    InputExample(guid=str(i), text_a=x)
                    for i, x in enumerate(candidates)
                ]
                features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    label_list=["0", "1"],
                    max_length=max_length,
                    output_mode="classification")
                # Convert to Tensors and build dataset
                all_input_ids = torch.tensor([f.input_ids for f in features],
                                             dtype=torch.long)
                all_attention_mask = torch.tensor(
                    [f.attention_mask for f in features], dtype=torch.long)
                all_labels = torch.tensor([0 for f in features], dtype=torch.long)
                all_token_type_ids = torch.tensor([[0.0] * max_length
                                                   for f in features],
                                                  dtype=torch.long)
                dataset = torch.utils.data.TensorDataset(all_input_ids,
                                                         all_attention_mask,
                                                         all_token_type_ids,
                                                         all_labels)
                return dataset

            eval_dataset = load_and_cache_examples(candidates, tokenizer)
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                sampler=torch.utils.data.SequentialSampler(eval_dataset),
                batch_size=max(1, torch.cuda.device_count()))
            preds = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[3]
                    }
                    if model_name.split('-')[0] != 'distilbert':
                        inputs['token_type_ids'] = batch[2] if model_name.split(
                            '-'
                        )[0] in [
                            'bert', 'xlnet'
                        ] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            return preds[:, 1].tolist()

        def convert_sentence_score_to_paragraph_score(sentence_score, sent_length):
            paragraph_score = []
            pointer = 0
            for i in sent_length:
                if i == 0:
                    paragraph_score.append(0.0)
                    continue
                temp_a = sentence_score[pointer:pointer + i]
                paragraph_score.append(sum(temp_a) / len(temp_a))
                pointer += i
            return paragraph_score

        model_name = 'bert-base-cased'
        saved_pretrained_CoLA_model_dir = './cola_model/' + model_name + '/'
        tokenizer, model = load_pretrained_cola_model(
            model_name, saved_pretrained_CoLA_model_dir)
        candidates = [y for x in sentences for y in x]
        sent_length = [len(x) for x in sentences]
        cola_score = evaluate_cola(model, candidates, tokenizer, model_name)
        cola_score = convert_sentence_score_to_paragraph_score(
            cola_score, sent_length)
        return cola_score


    def get_grammaticality_score(self, processed_candidates):
        lm_score = self.get_lm_score(processed_candidates)
        cola_score = self.get_cola_score(processed_candidates)
        grammaticality_score = [
            1.0 * math.exp(-0.5 * x) + 1.0 * y
            for x, y in zip(lm_score, cola_score)
        ]
        grammaticality_score = [
            max(0, x / 8.0 + 0.5) for x in grammaticality_score
        ]  # re-scale
        return grammaticality_score


    def get_redundancy_score(self,all_summary):

        def if_two_sentence_redundant(a, b):
            """ Determine whether there is redundancy between two sentences. """
            if a == b:
                return 4
            if (a in b) or (b in a):
                return 4
            flag_num = 0
            a_split = a.split()
            b_split = b.split()
            if max(len(a_split), len(b_split)) >= 5:
                longest_common_substring = difflib.SequenceMatcher(
                    None, a, b).find_longest_match(0, len(a), 0, len(b))
                LCS_string_length = longest_common_substring.size
                if LCS_string_length > 0.8 * min(len(a), len(b)):
                    flag_num += 1
                LCS_word_length = len(a[longest_common_substring[0]:(
                    longest_common_substring[0] +
                    LCS_string_length)].strip().split())
                if LCS_word_length > 0.8 * min(len(a_split), len(b_split)):
                    flag_num += 1
                edit_distance = editdistance.eval(a, b)
                if edit_distance < 0.6 * max(
                        len(a), len(b)
                ):  # Number of modifications from the longer sentence is too small.
                    flag_num += 1
                number_of_common_word = len([x for x in a_split if x in b_split])
                if number_of_common_word > 0.8 * min(len(a_split), len(b_split)):
                    flag_num += 1
            return flag_num

        redundancy_score = [0.0 for x in range(len(all_summary))]
        for i in range(len(all_summary)):
            flag = 0
            summary = all_summary[i]
            if len(summary) == 1:
                continue
            for j in range(len(summary) - 1):  # for pairwise redundancy
                for k in range(j + 1, len(summary)):
                    flag += if_two_sentence_redundant(summary[j].strip(),
                                                      summary[k].strip())
            redundancy_score[i] += -0.1 * flag
        return redundancy_score


    @Language.component("simhook")
    def SimilarityHook(doc):
#         return WMD.SpacySimilarityHook(doc)
        return doc


    def get_focus_score(self, all_summary):

        def compute_sentence_similarity():
            nlp = spacy.load('en_core_web_md')
            nlp.add_pipe('simhook', last=True)
            all_score = []
            for i in range(len(all_summary)):
                if len(all_summary[i]) == 1:
                    all_score.append([1.0])
                    continue
                score = []
                for j in range(1, len(all_summary[i])):
                    doc1 = nlp(all_summary[i][j - 1])
                    doc2 = nlp(all_summary[i][j])
                    try:
                        score.append(1.0 /
                                     (1.0 + math.exp(-doc1.similarity(doc2) + 7)))
                    except:
                        score.append(1.0)
                all_score.append(score)
            return all_score

        all_score = compute_sentence_similarity()
        focus_score = [0.0 for x in range(len(all_summary))]
        for i in range(len(all_score)):
            if len(all_score[i]) == 0:
                continue
            if min(all_score[i]) < 0.05:
                focus_score[i] -= 0.1
        return focus_score


    def get_gruen(self, candidates):
        processed_candidates = self.preprocess_candidates(candidates)
        grammaticality_score = self.get_grammaticality_score(processed_candidates)
        redundancy_score = self.get_redundancy_score(processed_candidates)
        focus_score = self.get_focus_score(processed_candidates)
        
        gruen_score = [
            min(1, max(0, sum(i)))
            for i in zip(grammaticality_score, redundancy_score, focus_score)
        ]
        return gruen_score
    
    def score(self, cs):
        scores = self.get_gruen(cs)
#         print(scores)
        return np.mean(scores)

##############################################################################################################################
##                                          MOVER SCORE                                                                     ##
##############################################################################################################################

class MoverScore:
    def __init__(self, use_gpu, gpu='cuda:0', n_gram=1):
        self.use_gpu   = use_gpu
        self.device = torch.device("cpu")
        self.n_gram = n_gram
        if(self.use_gpu):
            self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
            
    def sentence_score(self, hypothesis: str, references: List[str], trace=0):
        """Calculates the sentence score using Word Mover's Distance (WMD).

        Args:
            hypothesis (str): The hypothesis sentence.
            references (List[str]): A list of reference sentences.
            trace (int, optional): Controls verbosity (0: silent, 1: print details). Defaults to 0.

        Returns:
            float: The average WMD score between the hypothesis and each reference.
        """

        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        scores = word_mover_score(references, [hypothesis] * len(references), idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
        sentence_score = np.mean(scores)

        if trace > 0:
            print(hypothesis, references, sentence_score)

        return sentence_score

    def corpus_score(self, sys_stream: List[str], ref_stream: List[str], trace=0):
        """Calculates the corpus score by averaging sentence scores.

        Args:
            sys_stream (List[str]): A list of hypothesis sentences.
            ref_stream (List[str]): A list of reference sentences (one for each hypothesis).
            trace (int, optional): Controls verbosity (0: silent, 1: print details). Defaults to 0.

        Returns:
            float: The average sentence score across the corpus.
        """

        if len(sys_stream) != len(ref_stream):
            raise EOFError("Source and reference streams have different lengths!")

        corpus_scores = []
        for i in range(len(sys_stream)):
            hypo, ref = sys_stream[i], ref_stream[i]
            corpus_scores.append(self.sentence_score(hypo, [ref], trace=0))  # Send individual ref for each hypothesis

        return np.mean(corpus_scores)
    
    def score(self, cs, ref, trace = 0):
#         idf_dict_ref = get_idf_dict(ref)
#         idf_dict_hyp = get_idf_dict(cs)
#         scores = word_mover_score(ref, cs, 
#                           idf_dict_ref=idf_dict_ref, idf_dict_hyp=idf_dict_hyp, 
#                           stop_words=[], n_gram=1, remove_subwords=True)
#         return np.mean(scores)

        return self.corpus_score(cs, ref, trace)