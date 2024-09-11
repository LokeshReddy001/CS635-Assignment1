import os
import random
import wandb
import json
from dotenv import load_dotenv
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from models import BertClsFFN, BertLogitScorer
from utils import ContrastiveLoss


load_dotenv()
wandb.login(key=os.environ['WANDB_API_KEY'])


def train_DocLH_logits(dataset_path):
    task_name = 'DocLH_Logits'
    wandb.init(project=task_name)
    
    with open(dataset_path, 'r') as f:
        final_dict = json.load(f)

    random.shuffle(final_dict)
    final_dict = final_dict[:(50500)]

    train_samples = final_dict[:50000]
    dev_samples = final_dict[50000:]
    train_dataset = Dataset.from_list(train_samples)
    dev_dataset = Dataset.from_list(dev_samples)

    model = BertLogitScorer().to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_size = 8
    num_epochs = 2
    num_training_steps = (len(train_dataset)) * 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)

    criterion = ContrastiveLoss()

    save_dir = "./model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        for i in range(0, len(train_dataset), batch_size):
            batch_samples = [train_dataset[j] for j in range(i, min(i + batch_size, len(train_dataset)))]
            batch_texts = []
            for sample in batch_samples:
                texts = [sample['query'] + ' [SEP] ' + sample['positive_document']]
                texts += [sample['query'] + ' [SEP] ' + neg_doc for neg_doc in sample['negative_documents']]
                batch_texts.extend(texts)
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

            optimizer.zero_grad() 
            scores = model(inputs)

            batch_loss = 0
            for b in range(len(batch_samples)):
                start_idx = b * (1 + len(batch_samples[0]['negative_documents']))
                end_idx = start_idx + (1 + len(batch_samples[0]['negative_documents']))
                batch_loss += criterion(scores[start_idx:end_idx])

            batch_loss = batch_loss / len(batch_samples)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 
            wandb.log({'train_loss': batch_loss.item()})

            if i % 100 == 0:
                print(f"Step {i}/{len(train_dataset)} Loss: {batch_loss.item()}")

            if i % 800 == 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for i in range(0, len(dev_dataset)):
                        batch_texts = []
                        batch_texts += [dev_dataset[i]['query'] + ' [SEP] ' + dev_dataset[i]['positive_document']]
                        batch_texts += [dev_dataset[i]['query'] + ' [SEP] ' + neg_doc for neg_doc in dev_dataset[i]['negative_documents']]
                        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

                        scores = model(inputs)
                        batch_loss = criterion(scores)
                        total_loss += batch_loss.item()
                dev_loss = total_loss / (len(dev_dataset))
                print(f"Validation Loss at step {i}: {dev_loss}")
                wandb.log({'dev_loss': dev_loss})
                model.train()

        model_save_path = os.path.join(save_dir, f"{task_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


def train_QueryLH_logits(dataset_path):
    task_name = 'QueryLH_Logits'
    wandb.init(project=task_name)
    
    with open(dataset_path, 'r') as f:
        final_dict = json.load(f)

    random.shuffle(final_dict)
    final_dict = final_dict[:(50500)]

    train_samples = final_dict[:50000]
    dev_samples = final_dict[50000:]
    train_dataset = Dataset.from_list(train_samples)
    dev_dataset = Dataset.from_list(dev_samples)

    model = BertLogitScorer().to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_size = 8
    num_epochs = 1
    num_training_steps = (len(train_dataset)) * 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)

    criterion = ContrastiveLoss()

    save_dir = "./model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        for i in range(0, len(train_dataset), batch_size):
            batch_samples = [train_dataset[j] for j in range(i, min(i + batch_size, len(train_dataset)))]
            batch_texts = []
            for sample in batch_samples:
                texts = [sample['doc'] + ' [SEP] ' + sample['positive_query']]
                texts += [sample['doc'] + ' [SEP] ' + neg_que for neg_que in sample['negative_queries']]
                batch_texts.extend(texts)
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

            optimizer.zero_grad() 
            scores = model(inputs)

            batch_loss = 0
            for b in range(len(batch_samples)):
                start_idx = b * (1 + len(batch_samples[0]['negative_queries']))
                end_idx = start_idx + (1 + len(batch_samples[0]['negative_queries']))
                batch_loss += criterion(scores[start_idx:end_idx])

            batch_loss = batch_loss / len(batch_samples)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            scheduler.step() 
            wandb.log({'train_loss': batch_loss.item()})

            if i % 100 == 0:
                print(f"Step {i}/{len(train_dataset)} Loss: {batch_loss.item()}")

            if i % 800 == 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for i in range(0, len(dev_dataset)):
                        batch_texts = []
                        batch_texts += [dev_dataset[i]['doc'] + ' [SEP] ' + dev_dataset[i]['positive_query']]
                        batch_texts += [dev_dataset[i]['doc'] + ' [SEP] ' + neg_que for neg_que in dev_dataset[i]['negative_queries']]
                        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

                        scores = model(inputs)
                        batch_loss = criterion(scores)
                        total_loss += batch_loss.item()
                dev_loss = total_loss / (len(dev_dataset))
                print(f"Validation Loss at step {i}: {dev_loss}")
                wandb.log({'dev_loss': dev_loss})
                model.train()

        model_save_path = os.path.join(save_dir, f"{task_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


def train_DocLH_cls(dataset_path):
    task_name = 'DocLH_CLS'
    wandb.init(project=task_name)
    with open(dataset_path, 'r') as f:
        final_dict = json.load(f)

    random.shuffle(final_dict)
    final_dict = final_dict[:(50500)]

    train_samples = final_dict[:50000]
    dev_samples = final_dict[50000:]
    train_dataset = Dataset.from_list(train_samples)
    dev_dataset = Dataset.from_list(dev_samples)

    model = BertClsFFN().to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_size = 32
    num_epochs = 2
    num_training_steps = (len(train_dataset)) * num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=500, 
                                                num_training_steps=num_training_steps)
    criterion = ContrastiveLoss()

    save_dir = "./model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        for i in range(0, len(train_dataset), batch_size):
            batch_samples = [train_dataset[j] for j in range(i, min(i + batch_size, len(train_dataset)))]
            batch_texts = []
            for sample in batch_samples:
                texts = [sample['query'] + ' [SEP] ' + sample['positive_document']]
                texts += [sample['query'] + ' [SEP] ' + neg_doc for neg_doc in sample['negative_documents']]
                batch_texts.extend(texts)
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

            optimizer.zero_grad() 
            scores = model(inputs)

            batch_loss = 0
            for b in range(len(batch_samples)):
                start_idx = b * (1 + len(batch_samples[0]['negative_documents']))
                end_idx = start_idx + (1 + len(batch_samples[0]['negative_documents']))
                batch_loss += criterion(scores[start_idx:end_idx])

            batch_loss = batch_loss / len(batch_samples)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 
            wandb.log({'train_loss': batch_loss.item()})

            if i % 100 == 0:
                print(f"Step {i}/{len(train_dataset)} Loss: {batch_loss.item()}")

            if i % 800 == 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for i in range(0, len(dev_dataset)):
                        batch_texts = []
                        batch_texts += [dev_dataset[i]['query'] + ' [SEP] ' + dev_dataset[i]['positive_document']]
                        batch_texts += [dev_dataset[i]['query'] + ' [SEP] ' + neg_doc for neg_doc in dev_dataset[i]['negative_documents']]
                        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

                        scores = model(inputs)
                        batch_loss = criterion(scores)
                        total_loss += batch_loss.item()
                dev_loss = total_loss / (len(dev_dataset))
                print(f"Validation Loss at step {i}: {dev_loss}")
                wandb.log({'dev_loss': dev_loss})
                model.train()

                
     
        model_save_path = os.path.join(save_dir, f"{task_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    

def train_QueryLH_cls(dataset_path):
    task_name = 'QueryLH_CLS'
    wandb.init(project=task_name)
    with open(dataset_path, 'r') as f:
        final_dict = json.load(f)

    random.shuffle(final_dict)
    final_dict = final_dict[:(50500)]

    train_samples = final_dict[:50000]
    dev_samples = final_dict[50000:]
    train_dataset = Dataset.from_list(train_samples)
    dev_dataset = Dataset.from_list(dev_samples)

    model = BertClsFFN().to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_size = 32
    num_epochs = 2
    num_training_steps = (len(train_dataset)) * num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=500, 
                                                num_training_steps=num_training_steps)
    criterion = ContrastiveLoss()

    save_dir = "./model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        for i in range(0, len(train_dataset), batch_size):
            batch_samples = [train_dataset[j] for j in range(i, min(i + batch_size, len(train_dataset)))]
            batch_texts = []
            for sample in batch_samples:
                texts = [sample['doc'] + ' [SEP] ' + sample['positive_query']]
                texts += [sample['doc'] + ' [SEP] ' + neg_que for neg_que in sample['negative_queries']]
                batch_texts.extend(texts)
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

            optimizer.zero_grad() 
            scores = model(inputs)

            batch_loss = 0
            for b in range(len(batch_samples)):
                start_idx = b * (1 + len(batch_samples[0]['negative_queries']))
                end_idx = start_idx + (1 + len(batch_samples[0]['negative_queries']))
                batch_loss += criterion(scores[start_idx:end_idx])

            batch_loss = batch_loss / len(batch_samples)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 
            wandb.log({'train_loss': batch_loss.item()})

            if i % 100 == 0:
                print(f"Step {i}/{len(train_dataset)} Loss: {batch_loss.item()}")

            if i % 800 == 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for i in range(0, len(dev_dataset)):
                        batch_texts = []
                        batch_texts += [dev_dataset[i]['doc'] + ' [SEP] ' + dev_dataset[i]['positive_query']]
                        batch_texts += [dev_dataset[i]['doc'] + ' [SEP] ' + neg_que for neg_que in dev_dataset[i]['negative_queries']]
                        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

                        scores = model(inputs)
                        batch_loss = criterion(scores)
                        total_loss += batch_loss.item()
                dev_loss = total_loss / (len(dev_dataset))
                print(f"Validation Loss at step {i}: {dev_loss}")
                wandb.log({'dev_loss': dev_loss})
                model.train()
        
        model_save_path = os.path.join(save_dir, f"{task_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
def train(task_name, dataset_path):
    if task_name == 'DocLH_Logits':
        train_DocLH_logits(dataset_path)
    elif task_name == 'QueryLH_Logits':
        train_QueryLH_logits(dataset_path)
    elif task_name == 'DocLH_CLS':
        train_DocLH_cls(dataset_path)
    elif task_name == 'QueryLH_CLS':
        train_QueryLH_cls(dataset_path)
    else:
        print("Invalid task name")


def main():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--task_name", type=str, help="Name of the task")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")

    args = parser.parse_args()

    train(args.task_name, args.dataset_path)

if __name__ == "__main__":
    main()

