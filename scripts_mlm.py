from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

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
model_checkpoint = "vinai/bertweet-base"  # distilroberta-base

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, normalization=True, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    result = tokenizer(examples['text'], truncation=True,)
    return result

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)    #mlm

models = []
for i, raw_data in enumerate([Ironic_train_data, Non_ironic_train_data]): 

    for t in range(len(raw_data)):
        new_text = raw_data[t].replace("#USER#", "@user")
        raw_data[t] = new_text.replace("#URL#", "http:")

    datasets = Dataset.from_dict({'text':raw_data})
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"]) #

    # data = data_collator.torch_call([tokenized_datasets[i] for i in range(10)])
    # for i in range(10):
    #     print( tokenizer.decode( data['input_ids'][i].tolist() ) )


    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir="./results/%d/"%i,
        evaluation_strategy="no",
        save_strategy='no',
        per_device_train_batch_size=16,
        num_train_epochs=1.0,
        learning_rate=2e-5,
        weight_decay=0.00,
        lr_scheduler_type="constant", 
        gradient_accumulation_steps=2,
    ) # linear

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
batch_size = 8  # 200 is divisble by 8
with torch.no_grad():
    Ironic_model.eval()
    Non_ironic_model.eval()
    total_score = 0
    for user in valid_data.keys():
        texts = valid_data[user]
        user_label = valid_labels[user]

        for t in range(len(texts)):
            new_text = texts[t].replace("#USER#", "@user")
            texts[t] = new_text.replace("#URL#", "http:")

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
            # Flatten the tokens
            loss = loss_func(outputs['logits'].view(-1, outputs['logits'].size(-1)), inputs_batch['labels'].view(-1))
            log_prob_ironic = (-1) * loss
            # print(loss.sum()/(loss!=0).sum())
            # print(loss.view(shift_labels.size()))

            outputs = Non_ironic_model(input_ids=inputs_batch['input_ids'], attention_mask=inputs_batch['attention_mask'])
            # Flatten the tokens
            loss = loss_func(outputs['logits'].view(-1, outputs['logits'].size(-1)), inputs_batch['labels'].view(-1))
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
            
    print('accuracy: ', total_score / len(valid_data))



# =================== distilroberta-base =============

# per_device_train_batch_size=16,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 
# gradient_accumulation_steps=2,

# acc 0.78


# per_device_train_batch_size=16,
# num_train_epochs=1.0,
# learning_rate=2e-5,
# weight_decay=0.01,
# lr_scheduler_type="constant", 
# gradient_accumulation_steps=2,

# acc 0.76


# =================== vinai/bertweet-base =============

# per_device_train_batch_size=16,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 
# gradient_accumulation_steps=2,

# acc 0.66


# per_device_train_batch_size=16,
# num_train_epochs=1.0,
# learning_rate=2e-5,
# weight_decay=0.01,
# lr_scheduler_type="constant", 
# gradient_accumulation_steps=2,

# acc 0.79



# per_device_train_batch_size=16,
# num_train_epochs=1.0,
# learning_rate=2e-5,
# weight_decay=0.00,
# lr_scheduler_type="constant", 
# gradient_accumulation_steps=2,

# acc 0.75