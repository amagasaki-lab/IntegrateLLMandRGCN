import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
#from ignite.handlers import * # create_lr_scheduler_with_warmup のためにimport
from transformers import get_linear_schedule_with_warmup
import spacy
import clip

from datasets import load_dataset
from tqdm import tqdm

from rgcn_meta_ly3 import RgcnMetaModelKai2Mod1Ly3

from make_graph_Etype59 import MakeGraphKai5Etype59, MyTextGraphDatasetKai5
from make_graph_Etype59plus import MakeGraphKai5Etype59plus#, MyTextGraphDatasetKai

from load_datasets import QQP, PIT2015, MRPC

import pickle
from datetime import datetime
import math

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

def make_train_log(save_date, variant_model_name, dataset_name, batch_size, scheduler_name, lr, now_epoch, now_train_loss, now_test_loss):
    with open("./train_ablation_logs/train_log-" + save_date + ".txt", "a") as f:
        if now_epoch == 1:
            init_datalist = [f"variant_model_name:{variant_model_name}\n",f"dataset:{dataset_name}\n", f"batch_size:{batch_size}\n", f"lr:{lr}\n", f"scheduler:{scheduler_name}\n", "============\n"]
            f.writelines(init_datalist)
        datalist = [f"epoch:{now_epoch}\n", f"train_loss:{now_train_loss}\n", f"test_loss:{now_test_loss}\n", "\n"]
        f.writelines(datalist)

def load_dataset_for_test(variant_model_name, dataset_name, batch_size):
    # データの準備
    if dataset_name == "qqp":
        datasets_folder = "./datasets/"
        dataset_source = QQP(datasets_folder)
    elif dataset_name == "pit2015":
        datasets_folder = "./datasets/"
        dataset_source = PIT2015(datasets_folder)
    elif dataset_name == "mrpc":
        datasets_folder = ""
        dataset_source = MRPC(datasets_folder)
    print(f"loaded dataset {dataset_name}")
    
    # グラフデータの生成と追加
    graph_data_maker = None
    dataset_tag = ""
    mod_tag = ""
    if "Etype59plus" in variant_model_name:
        graph_data_maker = MakeGraphKai5Etype59plus()
        dataset_tag = "-Etype59plus"
        mod_tag = "5"
    elif "Etype59" in variant_model_name:
        graph_data_maker = MakeGraphKai5Etype59()
        dataset_tag = "-Etype59"
        mod_tag = "5"

    print(f"dataset_name:{dataset_name}, {dataset_tag}-kai{mod_tag}")



    with open(f"./pickle_datas/{dataset_name}{dataset_tag}-kai{mod_tag}-test.pickle", mode='br') as fi:
        dataset_test = pickle.load(fi)
    print(f"loaded pickle:{dataset_name}{dataset_tag}-kai{mod_tag}-test.pickle")
                

    # データローダーの作成
    custom_dataset_test = MyTextGraphDatasetKai5(dataset_test)
    test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=False)

    return graph_data_maker, dataset_source, test_loader


def tester(device, variant_model_name, aggregation_mode, dataset_name, hyper_params, num_of_try):
    epoch_size = hyper_params["epoch_size"]
    batch_size = hyper_params["batch_size"]
    lr = hyper_params["lr"]

    graph_data_maker, dataset_source, test_loader = load_dataset_for_test(variant_model_name, dataset_name, batch_size)
    header = dataset_source.get_header()

    # モデルのロード
    # モデルファイル名がKai5でもクラスはRgcnMetaModelKai2Mod1Ly3!!
    model = RgcnMetaModelKai2Mod1Ly3(device, graph_data_maker.len_deprel(), aggregation_mode).to(device)
    model_names = {
        "RgcnMetaKai5Mod1Etype59plusLy3":{
            "mrpc":{
                "concat-asub":[
                ]
            },
            "pit2015":{
                "concat-asub":[
                ]
            },
            "qqp":{
                "concat-asub":[
                ]
            }
        },
        "RgcnMetaKai5Mod1Etype59Ly3":{
            "mrpc":{
                "concat-asub":[
                ]
            },
            "pit2015":{
                "concat-asub":[
                ]
            },
            "qqp":{
                "concat-asub":[
                ]
            }
        }
    }
    model_name = model_names[variant_model_name][dataset_name][aggregation_mode][num_of_try]
    model.load_state_dict(torch.load(f"./output_models/{model_name}.pth"))

    for param in model.parameters():
        param.requires_grad = False

    # 検証
    model.eval()
    all_labels = []
    our_prediction = []
    with torch.no_grad():
        for single_test_data in test_loader:
            sentence1 = single_test_data[header[0]]
            sentence2 = single_test_data[header[1]]
            graph_data = single_test_data["graphdata"]
            # mini_sentences1 = single_test_data["mini_sentences1"]
            # mini_sentences2 = single_test_data["mini_sentences2"]
            N = single_test_data["num_of_nodes_sentence1"]
            #print("mini_sentences1 in batch roop:", mini_sentences1)
            
            labels = torch.tensor([round(float(s)) for s in single_test_data[header[2]]]).to(device)
            all_labels += labels
            outputs = model(graph_data, N)

            tmp_pred = []
            for pred in outputs:
                max_value = max(pred)
                max_index = pred.tolist().index(max_value)
                tmp_pred.append(max_index)
            our_prediction += tmp_pred

        #モデルとデータセット名を出力
        print(f"dataset_name:{dataset_name}")
        print(f"variant:{variant_model_name}")
        print(f"model_name:{model_name}")

        # 混同行列の計算
        cm = confusion_matrix(torch.tensor(all_labels), torch.tensor(our_prediction))
        print("Confusion Matrix:")
        print(cm)

        # 評価指標の計算
        accuracy = accuracy_score(torch.tensor(all_labels), torch.tensor(our_prediction))
        f1 = f1_score(torch.tensor(all_labels), torch.tensor(our_prediction))
        precision = precision_score(torch.tensor(all_labels), torch.tensor(our_prediction))
        recall = recall_score(torch.tensor(all_labels), torch.tensor(our_prediction))

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        print("\n classification_report")
        print(classification_report(torch.tensor(all_labels), torch.tensor(our_prediction)))

if __name__ == "__main__":
    print("PyTorch ==", torch.__version__)
    print("CUDA available", torch.cuda.is_available())
    print("CUDA ==", torch.version.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    variant_model_names = [
        "RgcnMetaKai5Mod1Etype59plusLy3",#========Etype59plus[0]
        "RgcnMetaKai5Mod1Etype59Ly3"#============Etype59     [1]
    ]
    variant_model_name = variant_model_names[1]
    aggregation_modes = ["concat", "concat-asub", "asub", "asub-mul"]
    aggregation_mode = aggregation_modes[1]

    hyper_params = {
        "epoch_size" : 4,
        "batch_size" : 16,
        "lr" : 1e-4
    }
    dataset_names = ["qqp", "pit2015", "mrpc"]
    dataset_name = dataset_names[2]
    
    print("==variant_model_name==")
    print(variant_model_name)
    print("==dataset_name==")
    print(dataset_name)
    print("==setted hyper params==")
    print(f"hyper_params:{hyper_params}")

    num_of_model = 1
    for i in range(num_of_model):
        print(f"===take {i}:START===")
        tester(device, variant_model_name, aggregation_mode, dataset_name, hyper_params, i)
        print(f"===take {i}:FINISH===\n\n")
