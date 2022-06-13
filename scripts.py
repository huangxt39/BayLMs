from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoConfig, AutoModel

import torch

import glob
from xml.etree import ElementTree as ET
import os
import random
import pickle


# import wandb
# wandb.init(project="my-test-project", entity="huangxt233")


from datasets.utils import disable_progress_bar
disable_progress_bar()

train_users = 160
valid_users = 50
splitted_data_path = './splitted_data'

if not os.path.exists(splitted_data_path):

    GT = "./pan22-author-profiling-training-2022-03-29/en/truth.txt"
    true_values = {}
    f=open(GT, encoding='utf-8')
    for line in f:
        linev = line.strip().split(":::")
        true_values[linev[0]] = linev[1]
    f.close()


    I=[]
    NI=[]
    for key in true_values.keys():
        if true_values[key] == "I":
            I.append(key)
        else:
            NI.append(key)
    print(len(I), len(NI))
    assert len(I) == len(NI)
    assert train_users + valid_users == len(I)

    
    random.shuffle(I)
    random.shuffle(NI)
    train_labels = {}
    valid_labels = {}
    for i in range(train_users):
        train_labels[I[i]] = "I"
        train_labels[NI[i]] = "NI"

    for i in range(train_users, train_users+valid_users):
        valid_labels[I[i]] = "I"
        valid_labels[NI[i]] = "NI"



    Ironic_train_data = []
    Non_ironic_train_data = []
    valid_data = {}
    

    for FILE in glob.glob("./pan22-author-profiling-training-2022-03-29/en/*.xml"):
        #The split command below gets just the file name,
        #without the whole address. The last slicing part [:-4]
        #removes .xml from the name, so that to get the user code

        parsedtree = ET.parse(FILE)
        documents = parsedtree.iter("document")

        texts = []
        for doc in documents:
            texts.append(doc.text)

        USERCODE = FILE.split("/")[-1][:-4]
        if USERCODE in train_labels.keys():
            if train_labels[USERCODE] == "I":
                Ironic_train_data.extend(texts)
            else:
                Non_ironic_train_data.extend(texts)
        else:
            assert USERCODE in valid_labels.keys()
            valid_data[USERCODE] = texts


    print(len(Ironic_train_data))
    print(len(Non_ironic_train_data))
    print(len(valid_data))

    splitted_data = [Ironic_train_data, Non_ironic_train_data, valid_data, valid_labels]
    pickle.dump(splitted_data, open(splitted_data_path, 'wb'), -1)

else:
    splitted_data = pickle.load(open(splitted_data_path, 'rb'))
    Ironic_train_data, Non_ironic_train_data, valid_data, valid_labels = splitted_data


# start training
model_checkpoint = "distilgpt2" #distilgpt2

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    result = tokenizer(examples['text'], truncation=True,)
    return result

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

models = []
for i, raw_data in enumerate([Ironic_train_data, Non_ironic_train_data]): 
    datasets = Dataset.from_dict({'text':raw_data})
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"]) #

    # data = data_collator.torch_call([tokenized_datasets[i] for i in range(10)])
    # for i in range(10):
    #     print( tokenizer.decode( data['input_ids'][i].tolist() ) )


    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    # config = AutoConfig.from_pretrained(model_checkpoint)
    # model = AutoModelForCausalLM.from_config(config)

    training_args = TrainingArguments(
        output_dir="./results/%d/"%i,
        evaluation_strategy="no",
        save_strategy='no',
        per_device_train_batch_size=8,
        num_train_epochs=0.5,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="constant", 
    ) # linear gradient_accumulation_steps=20,

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model()
    models.append(trainer.model)


# start evaluate
Ironic_model, Non_ironic_model = models
assert not (Ironic_model is Non_ironic_model)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')
batch_size = 8
with torch.no_grad():
    Ironic_model.eval()
    Non_ironic_model.eval()
    total_score = 0
    for user in valid_data.keys():
        texts = valid_data[user]
        user_label = valid_labels[user]
        user_tweets = Dataset.from_dict({'text':texts})
        tokenized_tweets = user_tweets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
        log_p_p = 0
        for i in range(0, len(tokenized_tweets), batch_size):
            end = i+batch_size
            inputs_batch = data_collator.torch_call([tokenized_tweets[j] for j in range(i,end)])
            # dict_keys(['input_ids', 'attention_mask', 'labels'])
            # outputs = model(**inputs_batch)
            # print(outputs['loss'])
            if torch.cuda.is_available():
                for key in inputs_batch.keys():
                    inputs_batch[key] = inputs_batch[key].cuda()


            outputs = Ironic_model(input_ids=inputs_batch['input_ids'], attention_mask=inputs_batch['attention_mask'])
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = inputs_batch['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            log_prob_ironic = (-1) * loss
            # print(loss.sum()/(loss!=0).sum())
            # print(loss.view(shift_labels.size()))

            outputs = Non_ironic_model(input_ids=inputs_batch['input_ids'], attention_mask=inputs_batch['attention_mask'])
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = inputs_batch['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            log_prob_non_ironic = (-1) * loss

            log_p_p += (log_prob_ironic - log_prob_non_ironic).sum()
        
        if log_p_p > 0:
            # ironic
            prediction = "I"
        else:
            # non ironic
            prediction = "NI"

        if prediction == user_label:
            total_score += 1
        else:
            print(prediction)
            
    print('accuracy: ', total_score / len(valid_data))

# ===================== distilgpt2 =======================

# per_device_train_batch_size=8,
# num_train_epochs=1.0,
# learning_rate=2e-5,
# weight_decay=0.01,
# lr_scheduler_type="constant",

# first final loss 3.755
# second final loss 4.0388
# acc 0.91



# per_device_train_batch_size=8,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 

# first final loss 3.7873
# second final loss 4.0295
# acc 0.92



# per_device_train_batch_size=32,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 

# first final loss 3.7286
# second final loss 4.0011
# acc 0.91

# using same parameters as above, with gradient_accumulation_steps=1000,
# acc 0.73

# using same parameters as above, with gradient_accumulation_steps=200,
# acc 0.83

# using same parameters as above, with gradient_accumulation_steps=50,
# acc 0.89

# using same parameters as above, with per_device_train_batch_size=8, gradient_accumulation_steps=20,
# acc 0.91



# config = AutoConfig.from_pretrained(model_checkpoint)
# model = AutoModelForCausalLM.from_config(config)
# per_device_train_batch_size=32,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 

# first final loss 5.1699
# second final loss 5.6563
# acc 0.91

# using same parameters as above, with gradient_accumulation_steps=1000,
# acc 0.53




# per_device_train_batch_size=8,
# num_train_epochs=3.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="linear", 

# first final loss 3.2173
# second final loss 3.4421
# acc 0.88



# per_device_train_batch_size=8,
# num_train_epochs=6.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="linear", 

# first final loss 2.7697
# second final loss 2.9811
# acc 0.74




# ===================== gpt2 =======================

# per_device_train_batch_size=8,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 
# gradient_accumulation_steps=4,

# acc 0.91


# try gpt2 bertweet
# try different epoch number , batch_size and we re done