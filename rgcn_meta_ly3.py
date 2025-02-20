# 2024/11/05 __split_and_pool()を修正．
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, Sequential
from torch_geometric.nn import global_mean_pool
from transformers import BertTokenizer, BertModel
import numpy as np
import spacy
import clip

# designed as CrossEncoder use RGCN-only
class RgcnMetaModelKai2Mod1Ly3(torch.nn.Module):
    def __init__(self, device, num_relations, aggregation_mode):
        super(RgcnMetaModelKai2Mod1Ly3, self).__init__()
        self.device = device
        
        self.node_features_dim = 768
        self.gcn_out_dim = 768
        self.hidden_dim = 512 # from SBERT-RGCN paper
        self.num_relations = num_relations

        self.aggregation_mode = aggregation_mode
        self.gcn_agg_out_dim = {"concat":2, "concat-asub":3, "asub":1, "asub-mul":2}
        if self.aggregation_mode not in self.gcn_agg_out_dim.keys():
            self.aggregation_mode = "concat"

        self.pooled_check_flag = True

        # define Networks
        #RGCN
        self.rgcn_conv1 = RGCNConv(self.node_features_dim, self.hidden_dim, self.num_relations)
        self.rgcn_conv2 = RGCNConv(self.hidden_dim, self.gcn_out_dim, self.num_relations)

        # # Meta-Model
        print("!!rgcn-meta-kai-mod1!!")
        print("!!USE call __split_and_pool!!")
        print(f"!!aggregation strategy : {self.aggregation_mode}!!")

        self.meta_fc = nn.Sequential(
            nn.Linear(self.gcn_out_dim * self.gcn_agg_out_dim[self.aggregation_mode], 128),# from SBERT-RGCN paper
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, graph_data, N):
        node_fts = graph_data.x.clone().detach().to(self.device)
        edge_index = graph_data.edge_index.clone().detach().to(self.device)
        edge_attr = graph_data.edge_attr.clone().detach().to(self.device)
        graph_batch = graph_data.batch.clone().detach().to(self.device)
        # RGCNから特徴を得る
        x = self.rgcn_conv1(node_fts, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.rgcn_conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        graph_embedding1, graph_embedding2 = self.__split_and_pool(x, graph_batch, N)
        
        # 出力の統合
        a_sub_emb = torch.abs(torch.sub(graph_embedding1, graph_embedding2))
        if self.aggregation_mode == "concat-asub":
            combined = torch.cat((graph_embedding1, graph_embedding2, a_sub_emb), dim=1)
        elif self.aggregation_mode == "asub":
            combined = a_sub_emb
        elif self.aggregation_mode == "asub-mul":
            mul_emb = graph_embedding1 * graph_embedding2
            combined = torch.cat((a_sub_emb, mul_emb), dim=1)
        else:# concat:
            combined = torch.cat((graph_embedding1, graph_embedding2), dim=1)

        # 最終的な出力
        output = self.meta_fc(combined)
        return output
    
    def __split_and_pool(self, x, graph_batch, N):
        all_front_pooled = []
        all_rear_pooled = []
        
        current_node_index = 0  # ノードインデックスの追跡
        
        # 各グラフごとに分割とpoolingを行う
        for i in range(len(N)):
            # グラフiの前半部分と後半部分のノード数
            num_nodes_front = N[i]
            num_nodes_total = (graph_batch == i).sum().item()  # グラフi全体のノード数
            num_nodes_rear = num_nodes_total - num_nodes_front  # 後半部分のノード数
            
            # 前半部分のノードインデックス (テンソル)
            front_nodes = torch.arange(current_node_index, current_node_index + num_nodes_front, device=x.device)
            # 後半部分のノードインデックス (テンソル)
            rear_nodes = torch.arange(current_node_index + num_nodes_front, current_node_index + num_nodes_total, device=x.device)
            
            # 後半部分の検出に失敗した場合
            if (current_node_index + num_nodes_front) == (current_node_index + num_nodes_total):
                print("!!! rear_nodes is NULL !!!")
                rear_nodes = front_nodes.clone()
            
            # 前半部分と後半部分のノード特徴量
            front_x = x[front_nodes]
            rear_x = x[rear_nodes]
            
            # global_mean_poolを適用
            front_pooled = torch.mean(input=front_x, dim=0)#列ごとの平均
            rear_pooled = torch.mean(input=rear_x, dim=0)

            if self.pooled_check_flag:
                self.pooled_check_flag = False
                print(f"front_pooled len:{len(front_pooled)} row[0:10]:{front_pooled[0:10]}")
                print(f" rear_pooled len:{len(rear_pooled)} row[0:10]:{rear_pooled[0:10]}")
            
            # 結果をリストに追加
            all_front_pooled.append(front_pooled)
            all_rear_pooled.append(rear_pooled)
            
            # ノードインデックスを更新
            current_node_index += num_nodes_total

        return torch.stack(all_front_pooled), torch.stack(all_rear_pooled)
        