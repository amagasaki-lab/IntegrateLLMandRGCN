import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, Sequential
from torch_geometric.nn import global_mean_pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
import spacy
import clip

# designed as CrossEncoder use RGCN-only
class CombineRgcnLy3DividePoolPromptEolModel(torch.nn.Module):
    def __init__(self, device, num_relations, aggregation_mode, base_model_name = "opt-2.7b"):
        super(CombineRgcnLy3DividePoolPromptEolModel, self).__init__()
        self.device = device
        
        self.node_features_dim = 768
        self.gcn_out_dim = 768
        self.hidden_dim = 512 # from SBERT-RGCN paper
        self.num_relations = num_relations

        self.aggregation_mode = aggregation_mode
        self.gcn_agg_out_dim = {"concat":2, "concat-asub":3, "asub":1, "asub-mul":2}
        if self.aggregation_mode not in self.gcn_agg_out_dim.keys():
            self.aggregation_mode = "concat"

        #RGCN
        self.rgcn_conv1 = RGCNConv(self.node_features_dim, self.hidden_dim, self.num_relations)
        self.rgcn_conv2 = RGCNConv(self.hidden_dim, self.gcn_out_dim, self.num_relations)
        
        #LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{base_model_name}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(f"facebook/{base_model_name}")#.to(self.device)
        self.llm_tokenizer.pad_token_id = 0 
        self.llm_tokenizer.padding_side = "left"

        self.peft_model = PeftModel.from_pretrained(self.llm_model, f"royokong/prompteol-{base_model_name}", torch_dtype=torch.float16)#.to(self.device)
        self.template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        self.peft_model_out_dim = 2560
        if base_model_name == "opt-1.3b":
            self.peft_model_out_dim = 2048
        print(f"base_model_name:{base_model_name}")

        # # Meta-Model
        self.meta_fc = nn.Sequential(
            nn.Linear(self.gcn_out_dim * self.gcn_agg_out_dim[self.aggregation_mode] + self.peft_model_out_dim * 2, 128),# from SBERT-RGCN paper
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        print("This is Combine(PromptEOL & RGCN-DividePool)Model.")
        print(f"!!aggregation strategy : {self.aggregation_mode}!!")

    def forward(self, sentence1s, sentence2s, graph_data, N):
        #rgcn
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

        # グラフ埋め込みの統合
        a_sub_emb = torch.abs(torch.sub(graph_embedding1, graph_embedding2))
        if self.aggregation_mode == "concat-asub":
            graph_embeddings = torch.cat((graph_embedding1, graph_embedding2, a_sub_emb), dim=1)
        elif self.aggregation_mode == "asub":
            graph_embeddings = a_sub_emb
        elif self.aggregation_mode == "asub-mul":
            mul_emb = graph_embedding1 * graph_embedding2
            graph_embeddings = torch.cat((a_sub_emb, mul_emb), dim=1)
        else:# concat:
            graph_embeddings = torch.cat((graph_embedding1, graph_embedding2), dim=1)
        
        #llm
        sentence1s_inputs = self.llm_tokenizer([self.template.replace('*sent_0*', i).replace('_', ' ') for i in sentence1s], padding=True, return_tensors="pt").to(self.device)
        sentence2s_inputs = self.llm_tokenizer([self.template.replace('*sent_0*', i).replace('_', ' ') for i in sentence2s], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding1s = self.peft_model(**sentence1s_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :].to(self.device)
            embedding2s = self.peft_model(**sentence2s_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :].to(self.device)
        llm_embedding1s_t = torch.stack([e.clone().detach() for e in embedding1s])
        llm_embedding2s_t = torch.stack([e.clone().detach() for e in embedding2s])
        llm_a_sub_emb = torch.abs(torch.sub(llm_embedding1s_t, llm_embedding2s_t))
        llm_mul_emb = llm_embedding1s_t * llm_embedding2s_t
        llm_embedding = torch.cat((llm_a_sub_emb, llm_mul_emb), dim=1)

        # 出力の統合
        combined = torch.cat((graph_embeddings, llm_embedding), dim=1)
        
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
                rear_nodes = front_nodes.clone().detach()
            
            # 前半部分と後半部分のノード特徴量
            front_x = x[front_nodes]
            rear_x = x[rear_nodes]
            
            # global_mean_poolを計算するためにバッチインデックスを前半と後半で別々に作成
            front_batch = graph_batch[front_nodes]
            rear_batch = graph_batch[rear_nodes]
            
            # global_mean_poolを適用
            front_pooled = global_mean_pool(front_x, front_batch)[0]
            rear_pooled = global_mean_pool(rear_x, rear_batch)[0]
            
            # 結果をリストに追加
            all_front_pooled.append(front_pooled)
            all_rear_pooled.append(rear_pooled)
            
            # ノードインデックスを更新
            current_node_index += num_nodes_total

        return torch.stack(all_front_pooled), torch.stack(all_rear_pooled)
        