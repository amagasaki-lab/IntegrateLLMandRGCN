# 2024/10/29 set self.data_path_dic ! plz update!
# 2024/10/30 fix MRPC() which use datasets lib
#from torch.utils.data import Dataset, DataLoader
import json
import sys
import os
import re
import math

class TsvDataLoaderKai:
    def __init__(self, dataset_name, data_path_dic, infile_header, skip_infile_header):
        self.dataset_name = dataset_name
        self.data_path_dic = data_path_dic
        self.header = ["sentence1", "sentence2", "label"]# 読み込み後はこれに統一
        self.infile_header = infile_header# 元のファイルのヘッダ
        self.skip_infile_header = skip_infile_header# bool

        self.data_train, self.data_dev, self.data_test = self.__load_data()

        print("loaded data length")
        print(f"self.data_train :{len(self.data_train)}")
        print(f"self.data_dev   :{len(self.data_dev)}")
        print(f"self.data_test  :{len(self.data_test)}")

    def __load_data(self):
        return_data = []
        for data_split in ["train", "dev", "test"]:
            tmp_data = []
            data_path = self.data_path_dic[data_split]
            infile = open(data_path, 'rt', encoding = "utf-8")
            for i, line in enumerate(infile):
                #line = line.decode('utf-8').strip()
                if line.startswith('-'): continue
                items = re.split("\t", line)

                if i == 0 and self.skip_infile_header:# 先頭行を飛ばす
                    print(f"self.skip_infile_header is {self.skip_infile_header}!")
                    print(f"skiped line is {items}")
                    continue

                label = items[self.infile_header["label"]]
                
                # sentence1 = re.split("\\s+",items[self.infile_header["sentence1"]].lower())#org
                # sentence2 = re.split("\\s+",items[self.infile_header["sentence2"]].lower())#org

                sentence1 = items[self.infile_header["sentence1"]]
                sentence2 = items[self.infile_header["sentence2"]]
                
                tmp_data.append({self.header[0]:sentence1, self.header[1]:sentence2, self.header[2]:label})
            infile.close()
            print(f"loaded {data_path}")
            return_data.append(tmp_data)
        return return_data

    # def __getitem__(self, idx):
    #     tmp_train_data = self.get_train_data()
    #     return tmp_train_data[idx]
    
    # def __len__(self):
    #     return len(self.data_train)

    def get_train_data(self):
        return self.data_train
    
    def get_dev_data(self):
        return self.data_dev

    def get_test_data(self):
        return self.data_test
    
    def get_header(self):
        return self.header
    
    def get_len_train(self):
        return len(self.data_train)
    
    def get_train_labels(self):
        train_labels = [ round(float(d[self.header[2]])) for d in self.data_train ]
        return train_labels


# QQP datast is from https://github.com/zhiguowang/BiMPM/tree/master?tab=readme-ov-file which referenced by SBERT-PAS paper.
class QQP(TsvDataLoaderKai):
    def __init__(self, cnt_dir = './'):
        super().__init__(
            dataset_name = "QQP",
            data_path_dic = {
                "train": cnt_dir + "Quora_question_pair_partition/train.tsv",
                "dev": cnt_dir + "Quora_question_pair_partition/dev.tsv",
                "test": cnt_dir + "Quora_question_pair_partition/test.tsv"
            },
            infile_header = {"sentence1":1, "sentence2":2, "label":0},
            skip_infile_header = False
        )

# PAWS-QQP dataset is generated with my PAWS_QQP.ipynb that based https://github.com/google-research-datasets/paws?tab=readme-ov-file
class PAWS_QQP(TsvDataLoaderKai):
    def __init__(self, cnt_dir = './'):
        super().__init__(
            dataset_name = "PAWS-QQP",
            data_path_dic = {
                "train": cnt_dir + "paws_qqp/train.tsv",
                "dev": cnt_dir + "paws_wiki_labeled_final/dev.tsv", #SBERT-PASに倣い，PAWS-Wikiのdevセットを代用
                #"dev": cnt_dir + "paws_qqp/dev_and_test.tsv",#
                "test": cnt_dir + "paws_qqp/dev_and_test.tsv"
            },
            infile_header = {"sentence1":1, "sentence2":2, "label":3},
            skip_infile_header = True
        )

# PAWS-Wiki datast is from https://github.com/google-research-datasets/paws?tab=readme-ov-file
class PAWS_Wiki(TsvDataLoaderKai):
    def __init__(self, cnt_dir = './'):
        super().__init__(
            dataset_name = "PAWS-Wiki",
            data_path_dic = {
                "train": cnt_dir + "paws_wiki_labeled_final/train.tsv",
                "dev": cnt_dir + "paws_wiki_labeled_final/dev.tsv",
                "test": cnt_dir + "paws_wiki_labeled_final/test.tsv"
            },
            infile_header = {"sentence1":1, "sentence2":2, "label":3},
            skip_infile_header = True
        )

# PIT2015 datast is from  https://github.com/cocoxu/SemEval-PIT2015/tree/master
class PIT2015(TsvDataLoaderKai):
    def __init__(self, cnt_dir = './'):
        super().__init__(
            dataset_name = "PIT2015",
            data_path_dic = {
                "train": cnt_dir + "pit2015/train.data",
                "dev": cnt_dir + "pit2015/dev.data",
                "test": cnt_dir + "pit2015/test.data",
                "test_label": cnt_dir + "pit2015/test.label"
            },
            infile_header = {"sentence1":2, "sentence2":3, "label":4},
            skip_infile_header = False
        )

        #無理やり読み直し
        self.data_train, self.data_dev, self.data_test = self.__reload_data()

        print("reloaded data length")
        print(f"self.data_train :{len(self.data_train)}")
        print(f"self.data_dev   :{len(self.data_dev)}")
        print(f"self.data_test  :{len(self.data_test)}")
    
    def __reload_data(self):# __init__()の中で呼びたかったけど断念
        print("__reload_data !")
        test_labels = self.__load_label_data()#testセットはラベルが別ファイルのため，別途読み込み

        return_data = []
        for data_split in ["train", "dev", "test"]:
            label_cnt = {"0":0, "1":0, "----":0, "(2, 3)":0} # 各ラベルについて，どの程度割り振られたかカウントし記録
            tmp_data = []
            data_path = self.data_path_dic[data_split]
            infile = open(data_path, 'rt', encoding = "utf-8")
            for i, line in enumerate(infile):
                if line.startswith('-'): continue
                items = re.split("\t", line)

                # ラベルをつける．提供元のREADMEに従い，言い換えか否かの意見が割れるものは除去．
                if data_split == "test":
                    label = test_labels[i]
                    if label == "----":
                        label_cnt[label] += 1
                        continue #データから除外（testセットの場合）
                else:
                    label_str_tuple = items[self.infile_header["label"]]
                    if label_str_tuple in ["(3, 2)", "(4, 1)", "(5, 0)"]:
                        label = "1"
                    elif label_str_tuple == "(2, 3)":
                        label_cnt["(2, 3)"] += 1
                        continue #データから除外（trainもしくはdevセットの場合）
                    else:
                        label = "0"
                label_cnt[label] += 1

                sentence1 = items[self.infile_header["sentence1"]]
                sentence2 = items[self.infile_header["sentence2"]]
                
                tmp_data.append({self.header[0]:sentence1, self.header[1]:sentence2, self.header[2]:label})
            infile.close()
            print(f"loaded {data_path}")
            print(f"label_cnt:{label_cnt}")
            return_data.append(tmp_data)
        
        return return_data
    
    def __load_label_data(self):
        labels = []
        data_path = self.data_path_dic["test_label"]
        infile = open(data_path, 'rt', encoding = "utf-8")
        for i, line in enumerate(infile):
            items = re.split("\t", line)

            tmp_label = items[0]
            if tmp_label == "false":
                labels.append("0")
            elif tmp_label == "true":
                labels.append("1")
            else:
                labels.append("----")
        infile.close()

        return labels

from datasets import load_dataset
class MRPC: #他と違い，ローカルのTSVファイルではなく，load_datasetから読み込む
    def __init__(self, cnt_dir = ""):
        self.dataset_name = "MRPC"
        self.header = ["sentence1", "sentence2", "label"]# 読み込み後はこれに統一

        self.dataset_org = load_dataset("nyu-mll/glue", "mrpc")
        self.data_train = self.dataset_org["train"]
        self.data_dev = self.dataset_org["validation"]
        self.data_test = self.dataset_org["test"]

        print("loaded data length")
        print(f"self.data_train :{len(self.data_train)}")
        print(f"self.data_dev   :{len(self.data_dev)}")
        print(f"self.data_test  :{len(self.data_test)}")
    
    def get_train_data(self):
        return self.data_train
    
    def get_dev_data(self):
        return self.data_dev

    def get_test_data(self):
        return self.data_test
    
    def get_header(self):
        return self.header
    
    def get_len_train(self):
        return len(self.data_train)