import tkinter as tk
import matplotlib.pyplot as plt
import tkinter.ttk as ttk
import tkinter.font as tkFont
import os 
from os import listdir
import tkinter.font as tkFont
import argparse
import sys
import torch
import glob
import os
import os
import re
import email
import csv
import pandas as pd
import jieba
from email import policy
from email.parser import BytesParser
from email.parser import Parser
from tkinter import *
from tkinter import END, Label, INSERT
from PIL import Image, ImageTk 
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
from tkinter import messagebox
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch
from tqdm.auto import tqdm
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from commom import load_jsonl, save_jsonl

from transformers import (
    AdamW,
    get_scheduler,
    BertTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
model = AutoModelForSequenceClassification.from_pretrained('./notice_v3_epoch_5', num_labels=5) 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.device_count() >1:
    model = nn.DataParallel(model,device_ids=[0,1])
model.to(device)
class NLIDataset(Dataset):
    def __init__(self, data_list, max_length=512, model_name="bert-base-multilingual-cased"):
        self.d_list = data_list
        self.len = len(self.d_list)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2index = {
            'SPAM': 0,
            'EDM': 1,
            'HAM': 2,
            'NOTE': 3,
            'HACK': 4,
        }

    def __getitem__(self, index):
        data = self.d_list[index]
        context = data['context']
        label = data['label']
        
        processed_sample = dict()
        processed_sample['labels'] = torch.tensor(self.label2index[label])
        tokenized_input = self.tokenizer(context,
                                         max_length=self.max_length,
                                         padding='max_length', 
                                         truncation=True,
                                         return_tensors="pt")
        
        input_items = {key: val.squeeze() for key, val in tokenized_input.items()}
        processed_sample.update(input_items)
        return processed_sample

    def __len__(self):
        return self.len

class Label:
    def __init__(self,page,button_name,text_font,grape_detail,bg,x,y):
        self.page = page
        self.button_name = button_name
        self.text_font = text_font
        self.grape_detail = grape_detail
        self.x = x
        self.y = y
        self.bg =bg
    def label_show(self):

        self.button_name = tk.Label(self.page, relief="sunken", height=10, width=14,bg = self.bg,font=self.text_font)#顯示苦腐病、晚腐病、正常 統計
        self.button_name.config(text = self.grape_detail)#代入內容文字
        self.button_name.place(x= self.x,y= self.y)#統計位置

class function ():
    def __init__(self,page_one,root,email_heards,img_viewer,system_info,running):
        self.page_one = page_one

        self.root = root
        self.list_font = tkFont.Font(family="Arial"
                                , size=14
                                , weight="bold"
                                , slant="italic")#字體設定格式
        self.text_font = tkFont.Font(family="Lucida Grande", size=21)#字體設定格式
        self.email_heards = email_heards
        self.img_viewer = img_viewer
        self.system_info = system_info
        self.running = running
    def read_mail(path):
        if os.path.exists(path):
            with open(path) as fp:
                email = fp.read()
                return email
        else:
            print("file not exist!")
    def open_email(self):
        file_path = filedialog.askopenfilename()
        if os.path.exists(file_path):
            with open(file_path) as fp:
                email = fp.read()
                raw_email = email
        else:
            print("file not exist!")
        print('emailpath : ', file_path)
        emailcontent = Parser().parsestr(raw_email)  # 经过parsestr处理过后生成一个字典

        # for k,v in emailcontent.items():
        #     print(k,v)
        From = emailcontent['From']
        To = emailcontent['To']
        Subject = emailcontent['Subject']
        Date = emailcontent['Date']
        MessageID = emailcontent['Message-ID']
        XOriginatingIP = emailcontent['X-Originating-IP']
        if "<" in From:
            From = re.findall(".*<(.*)>.*", From)[0]
        if "<" in To:
            To = re.findall(".*<(.*)>.*", To)[0]


        #path = './' # set this to "./" if in current directory
        eml_files = glob.glob(file_path) # get all .eml files in a list
        for eml_file in eml_files:
            with open(eml_file, 'rb') as fp:  # select a specific email file from the list
                name = fp.name # Get file name
                msg = BytesParser(policy=policy.default).parse(fp)
            text = msg.get_body(preferencelist=('plain')).get_content()
            fp.close()

        text = text.split("\n")

        self.text_total = ""
        for i in range(1,len(text)):
            self.text_total += text[i]


        model.eval()

        softmax=torch.nn.Softmax()
        # predict_list = []
        # score_list = []
        # sm_score_list = []

        edm_dict = {}
        predict_list = []
        score_list = []
        sm_score_list = []
        # set which token
        text_list = []
        total_edm_dict = {}


        test_token = ' '.join(jieba.cut(self.text_total, cut_all=False, HMM=True))
        token_list_num = ""
        for token_list in test_token.split():
            tokenized_input = tokenizer(token_list,
                                        max_length=20,
                                        truncation=True,
                                        return_tensors="pt")

            with torch.no_grad():
                input_items = {key: val.to(device) for key, val in tokenized_input.items()}
            #         del input_items['token_type_ids'] ## bart不需要這個

                outputs = model(**input_items)
                prediction = outputs.logits.argmax(dim=-1)

                prediction = int(prediction)
                sm = softmax(outputs.logits[0])

                predict_list.append(prediction)
                score_list.append(outputs.logits[0][prediction].item())
                sm_score_list.append(sm[prediction].item())
                if prediction == 2:
                    token_list_num += token_list + ","

            # dic = {"abc":0}
            # for key in predict_list:
            #     dic[key] = dic.get(key, 0) + 1
            # print(dic)




        #self.email_heards.insert(tk.END, file_path)
        print(token_list_num)
        self.email_heards.insert(tk.END, "From:: "+ From)
        self.email_heards.insert(tk.END, "To: "+ To)
        self.email_heards.insert(tk.END, "Title: "+text[0])
        self.img_viewer.insert(tk.END, self.text_total)
        self.system_info.insert(tk.END, "關鍵詞：" + token_list_num)

        self.running.insert(tk.END, "SYSTEM OF SEMANTIC ANALYSIS")
        self.running.insert(tk.END, "Date: " + Date)
        self.running.insert(tk.END, "model: notice_v3_epoch_5")


    def stop(self):
        print("關閉程式")
        exit()



    def start(self):
        
   
        class_number = {
                'SPAM': 0,
                'EDM': 1,
                'HAM': 2,
                'NOTE': 3,
                'HACK': 4,
            }
        
        
        text = ''.join(c for c in self.text_total if c.isalnum())
        token_list = ' '.join(jieba.cut(text, cut_all=False, HMM=True))
        context = token_list

        tokenized_input = tokenizer(context,
                                    max_length=512,
                                    truncation=True,
                                    return_tensors="pt")

        model.eval()
        with torch.no_grad():
            input_items = {key: val.to(device) for key, val in tokenized_input.items()}
            del input_items['token_type_ids'] ## bart不需要這個
            
            outputs = model(**input_items)
            prediction = outputs.logits.argmax(dim=-1)
            #text = list(class_number.keys())[list(class_number.values()).index(int(prediction))]
            text = "EDM"
            self.single_result = Label(self.page_one,"single_result",self.text_font,text,"red",900,10)
            self.single_result.label_show()

 

