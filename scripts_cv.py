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

I_user_per_fold = int(210 / 5)
data_folds_path = './data_folds'

if not os.path.exists(data_folds_path):

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

    
    random.shuffle(I)
    random.shuffle(NI)

    all_user_data = {}
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
        all_user_data[USERCODE] = texts


    user_folds = []
    for k in range(5):
        s = k*I_user_per_fold
        e = s + I_user_per_fold
        user_labels = {}
        for i in range(s, e):
            user_labels[I[i]] = "I"
            user_labels[NI[i]] = "NI"
        user_folds.append(user_labels)
        

    data_folds = [user_folds, all_user_data]
    pickle.dump(data_folds, open(data_folds_path, 'wb'), -1)

else:
    data_folds = pickle.load(open(data_folds_path, 'rb'))
    user_folds, all_user_data = data_folds

def merge_folds(train_users):
    result = {}
    for user_labels in train_users:
        result.update(user_labels)
    return result

acc_list = []
for k in range(5):

    train_users = []
    for i, user_labels in enumerate(user_folds):
        if i != k :
            train_users.append(user_labels)
        else:
            valid_labels = user_labels

    train_users = merge_folds(train_users)

    Ironic_train_data = []
    Non_ironic_train_data = []
    for USERCODE in train_users.keys():
        if train_users[USERCODE] == "I":
            Ironic_train_data.extend(all_user_data[USERCODE])
        else:
            Non_ironic_train_data.extend(all_user_data[USERCODE])
    print(len(Ironic_train_data))
    print(len(Non_ironic_train_data))

    valid_data = {}
    for USERCODE in valid_labels.keys():
        valid_data[USERCODE] = all_user_data[USERCODE]
    print(len(valid_data))

    # start training
    model_checkpoint = "gpt2" #distilgpt2

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
            num_train_epochs=1.0,
            learning_rate=1e-4,
            weight_decay=0.01,
            lr_scheduler_type="constant", 
        ) # linear gradient_accumulation_steps=2,

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
    batch_size = 2
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
                end = min(i+batch_size, len(tokenized_tweets))
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
                
        print('accuracy: ', total_score / len(valid_data))
        acc_list.append(total_score / len(valid_data))
        del Ironic_model
        del Non_ironic_model

print(acc_list)
print(sum(acc_list)/len(acc_list) )


# per_device_train_batch_size=8,
# num_train_epochs=1.0,
# learning_rate=1e-4,
# weight_decay=0.01,
# lr_scheduler_type="constant", 

# [0.9523809523809523, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.8809523809523809]
# 0.9166666666666666

# same as above, but with gpt2
# [0.9404761904761905, 0.8928571428571429, 0.9166666666666666, 0.9047619047619048, 0.8928571428571429]
# 0.9095238095238095


