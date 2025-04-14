import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer,MarianMTModel,Seq2SeqTrainingArguments,Seq2SeqTrainer,pipeline,DataCollatorForSeq2Seq


model_name='helsinki-NLP/opus-mt-en-fr'
epochs=3
batch_size=4

df=pd.read_csv("/content/drive/MyDrive/NLP/Data/translation_data.csv")
dataset=Dataset.from_pandas(df)


tokenizer=MarianTokenizer.from_pretrained(model_name)
model=MarianMTModel.from_pretrained(model_name)

def preprocess(example):
  tokenized_source=tokenizer(example["source"],max_length=128,truncation=True,padding="max_length")
  with tokenizer.as_target_tokenizer():
    tokenized_target=tokenizer(example["target"],max_length=128,truncation=True,padding="max_length")
  tokenized_source["labels"]=tokenized_target["input_ids"]
  return tokenized_source

tokenized_dataset=dataset.map(preprocess,batched=True)

training_arguments=Seq2SeqTrainingArguments(output_dir="./tmp",report_to="none",per_device_train_batch_size=batch_size,num_train_epochs=epochs)

data_collator=DataCollatorForSeq2Seq(model=model,tokenizer=tokenizer)

trainer=Seq2SeqTrainer(model=model,tokenizer=tokenizer,data_collator=data_collator,args=training_arguments,train_dataset=tokenized_dataset)

trainer.train()

translator=pipeline("translation",model=model,tokenizer=tokenizer)

translator("How are you?")[0]

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,pipeline

df=pd.read_csv("/content/drive/MyDrive/NLP/Data/sentiment_data.csv")
df['label']=df['label'].map({"negative":0,"positive":1})
dataset=Dataset.from_pandas(df)

model_name="distilbert-base-uncased"
batch_size=4
epochs=3

tokenizer=DistilBertTokenizerFast.from_pretrained(model_name)
model=DistilBertForSequenceClassification.from_pretrained(model_name)

def preprocess(example):
  tokenized_input=tokenizer(example["text"],max_length=256,truncation=True,padding="max_length")
  return tokenized_input


data_collator=DataCollatorWithPadding(tokenizer)

tokenized_dataset=dataset.map(preprocess,batched=True)
training_args=TrainingArguments(output_dir="./tmp",num_train_epochs=epochs,per_device_train_batch_size=batch_size,report_to='none')

trainer=Trainer(model=model,tokenizer=tokenizer,data_collator=data_collator,args=training_args,train_dataset=tokenized_dataset)
trainer.train()


sentiment_analyzer=pipeline("text-classification",model=model,tokenizer=tokenizer)
sentiment_analyzer("This is the worst book I’ve ever read.")

import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer,BartForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer,DataCollatorForSeq2Seq,pipeline


model_name='facebook/bart-base'
epochs=3
batch_size=4

df=pd.read_csv("/content/drive/MyDrive/NLP/Data/summarization_data.csv")
dataset=Dataset.from_pandas(df)

tokenizer=BartTokenizer.from_pretrained(model_name)
model=BartForConditionalGeneration.from_pretrained(model_name)

def preprocess(example):
  tokenized_input=tokenizer(example['text'],max_length=128,truncation=True,padding="max_length")
  with tokenizer.as_target_tokenizer():
    tokenized_summary=tokenizer(example['summary'],max_length=16,truncation=True,padding="max_length")
  tokenized_input['labels']=tokenized_summary['input_ids']
  return tokenized_input

tokenized_dataset=dataset.map(preprocess,batched=True)

data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)

training_args=Seq2SeqTrainingArguments(output_dir='./tmp',num_train_epochs=epochs,per_device_train_batch_size=batch_size,report_to="none")

trainer=Seq2SeqTrainer(model=model,tokenizer=tokenizer,args=training_args,data_collator=data_collator,train_dataset=tokenized_dataset)

trainer.train()

summarizer=pipeline("summarization",model=model,tokenizer=tokenizer,max_length=16)

summarizer("Machine translation is a sub-field of computational linguistics that is concerned with the use of computer software to translate text or speech from one language to another. It involves statistical and neural networks methods to improve translation quality.")

