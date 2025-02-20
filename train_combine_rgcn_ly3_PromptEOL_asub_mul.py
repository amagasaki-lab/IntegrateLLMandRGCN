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

from combine_rgcn_ly3_PromptEOL_asub_mul import CombineRgcnLy3DividePoolPromptEolModel

from make_graph_Etype59_fixPIT2015dev import MakeGraphKai5Etype59
from make_graph_Etype59plus_fixPIT2015dev import MakeGraphKai5Etype59plus, MyTextGraphDatasetKai5

from load_datasets_fixPIT2015dev import QQP, PAWS_QQP, PAWS_Wiki, PIT2015, MRPC

import pickle
from datetime import datetime
import math

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def make_train_log(save_date, variant_model_name, dataset_name, batch_size, scheduler_name, lr, now_epoch, now_train_loss, now_val_loss):
    with open("./train_logs/train_log-" + save_date + ".txt", "a") as f:
        if now_epoch == 1:
            init_datalist = [f"variant_model_name:{variant_model_name}\n",f"dataset:{dataset_name}\n", f"batch_size:{batch_size}\n", f"lr:{lr}\n", f"scheduler:{scheduler_name}\n", "============\n"]
            f.writelines(init_datalist)
        datalist = [f"epoch:{now_epoch}\n", f"train_loss:{now_train_loss}\n", f"val_loss:{now_val_loss}\n", "\n"]
        f.writelines(datalist)

def load_dataset_for_train(variant_model_name, dataset_name, batch_size):
    # データの準備
    if dataset_name == "qqp":
        datasets_folder = "./datasets/"
        dataset_source = QQP(datasets_folder)
    elif dataset_name == "pawsqqp":
        datasets_folder = "./datasets/"
        dataset_source = PAWS_QQP(datasets_folder)
    elif dataset_name == "pawswiki":
        datasets_folder = "./datasets/"
        dataset_source = PAWS_Wiki(datasets_folder)
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

    validation_split_tag = ""
    if dataset_name == "pit2015":
        validation_split_tag = "_Guidelined"
    print(f"validation_split_tag:{validation_split_tag}")

    print(f"dataset_name:{dataset_name}, {dataset_tag}-kai{mod_tag}")

    with open(f"./pickle_datas/{dataset_name}{dataset_tag}-kai{mod_tag}-train.pickle", mode='br') as fi:
        dataset_train = pickle.load(fi)
    with open(f"./pickle_datas/{dataset_name}{dataset_tag}-kai{mod_tag}-validation{validation_split_tag}.pickle", mode='br') as fi:
        dataset_validation = pickle.load(fi)
    if dataset_name == "pawsqqp":
        with open(f"./pickle_datas/{dataset_name}{dataset_tag}-kai{mod_tag}-test.pickle", mode='br') as fi:
            print(f"dataset is {dataset_name}, so reload test.pickle")
            dataset_validation = pickle.load(fi)
                

    # データローダーの作成
    custom_dataset_train = MyTextGraphDatasetKai5(dataset_train)
    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
    custom_dataset_validation = MyTextGraphDatasetKai5(dataset_validation)
    val_loader = DataLoader(custom_dataset_validation, batch_size=batch_size, shuffle=True)

    return graph_data_maker, dataset_source, train_loader, val_loader


def trainer(device, variant_model_name, aggregation_mode, save_name, dataset_name, hyper_params, base_model_name):
    epoch_size = hyper_params["epoch_size"]
    batch_size = hyper_params["batch_size"]
    lr = hyper_params["lr"]

    graph_data_maker, dataset_source, train_loader, val_loader = load_dataset_for_train(variant_model_name, dataset_name, batch_size)
    header = dataset_source.get_header()

    # モデルのロード
    model = CombineRgcnLy3DividePoolPromptEolModel(device, graph_data_maker.len_deprel(), aggregation_mode, base_model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    for name, param in model.named_parameters():
        if "rgcn_conv" in name or "meta_fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(f"{name}    requires_grad:{param.requires_grad}")
    optimizer = torch.optim.Adam([
        {'params': model.rgcn_conv1.parameters(), 'lr': lr},
        {'params': model.rgcn_conv2.parameters(), 'lr': lr},
        {'params': model.meta_fc.parameters(), 'lr': lr}
    ])
    best_val_loss = 999999.9

    scheduler_name = "get_linear_schedule_with_warmup" # inspired by SBERT-RGCN
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(dataset_source.get_len_train() / batch_size) * 0.1,
        num_training_steps=math.ceil(dataset_source.get_len_train() / batch_size) * epoch_size
    )
    print(f"scheduler:{scheduler_name}")
    
    # トレーニングループ 
    for epoch in range(epoch_size):
        model.train()
        print_flag = True
        train_loss = 0.0
        for single_train_data in train_loader:
            optimizer.zero_grad()
            sentence1 = single_train_data[header[0]]
            sentence2 = single_train_data[header[1]]
            graph_data = single_train_data["graphdata"]
            # mini_sentences1 = single_train_data["mini_sentences1"]
            # mini_sentences2 = single_train_data["mini_sentences2"]
            N = single_train_data["num_of_nodes_sentence1"]
            #print("mini_sentences1 in batch roop:", mini_sentences1)
            
            labels = torch.tensor([round(float(s)) for s in single_train_data[header[2]]]).to(device)
                #labels = torch.tensor([int(s) for s in single_train_data[header[2]]]).to(device)

            # print(f"sentence1s:{sentence1}")
            # exit()
            outputs = model(sentence1, sentence2, graph_data, N)

            if print_flag and epoch >= 0:
                print_flag = False
                print("len(outputs)", len(outputs))
                print("len(labels)", len(labels))
                print("outputs:", outputs)
                print("labels :", labels)
            loss = criterion(outputs, labels)
            if torch.isnan(loss): # Lossがnanになっていないか確認
                print("Loss is NaN")
            loss.backward()
            train_loss += loss.item()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# 勾配クリッピング（nan対策）
            optimizer.step()# パラメータの更新

            #print(f"scheduler.get_lr():{scheduler.get_lr()}")
            scheduler.step()# get_linear_schedule_with_warmupのとき
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}")

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for single_val_data in val_loader:
                sentence1 = single_val_data[header[0]]
                sentence2 = single_val_data[header[1]]
                graph_data = single_val_data["graphdata"]
                # mini_sentences1 = single_val_data["mini_sentences1"]
                # mini_sentences2 = single_val_data["mini_sentences2"]
                N = single_val_data["num_of_nodes_sentence1"]
                
                labels = torch.tensor([round(float(s)) for s in single_val_data[header[2]]]).to(device)
                    #labels = torch.tensor([int(s) for s in single_val_data[header[2]]]).to(device)
            
                outputs = model(sentence1, sentence2, graph_data, N)
                
                # print("outputs length ", len(outputs))
                # print("labels  length ", len(labels))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")
        make_train_log(save_date, variant_model_name, dataset_name, batch_size, scheduler_name, lr, epoch+1, train_loss/len(train_loader), val_loss/len(val_loader))
        if best_val_loss > val_loss/len(val_loader):
            best_val_loss = val_loss/len(val_loader)
            if dataset_name == "pit2015":
                tmp_savename = save_name + "_Guidelined_best"
            else:
                tmp_savename = save_name + "_best"
            torch.save(model.state_dict(), './output_ablation_models/' +  tmp_savename +'.pth')
            print(f"epoch:{epoch+1} saved {tmp_savename}")

if __name__ == "__main__":
    print("PyTorch ==", torch.__version__)
    print("CUDA available", torch.cuda.is_available())
    print("CUDA ==", torch.version.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    variant_model_names = [
        "CombineRgcnEtype59plusKai5Ly3DividePoolPromptEolModelAsubMul",#========Etype59plus[0]
        "CombineRgcnEtype59Kai5Ly3DividePoolPromptEolModelAsubMul",#============Etype59    [1]
    ]
    variant_model_name = variant_model_names[0]
    base_model_names = [
        "opt-2.7b",
        "opt-1.3b"
    ]
    base_model_name = base_model_names[0]
    aggregation_modes = ["concat", "concat-asub", "asub", "asub-mul"]
    aggregation_mode = aggregation_modes[1]

    hyper_params = {
        "epoch_size" : 4,
        "batch_size" : 16,
        "lr" : 1e-4
    }
    dataset_names = ["qqp", "pawsqqp", "pawswiki", "pit2015", "mrpc"]
    dataset_name = dataset_names[4]
    
    print("==variant_model_name==")
    print(variant_model_name)
    print("==PromptEOL base_medel_name")
    print(base_model_name)
    print("==dataset_name==")
    print(dataset_name)
    print("==setted hyper params==")
    print(f"hyper_params:{hyper_params}")

    num_of_model = 1
    for i in range(num_of_model):
        print(f"===take {i}:START===")
        save_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_name = variant_model_name + f"_{base_model_name}_" + aggregation_mode + "_model_trained_" + dataset_name + "-kai-" + save_date
        trainer(device, variant_model_name, aggregation_mode, save_name, dataset_name, hyper_params, base_model_name)
        print(f"===take {i}:FINISH===\n\n")
