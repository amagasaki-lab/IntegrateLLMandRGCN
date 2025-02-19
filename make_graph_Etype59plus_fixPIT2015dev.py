import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData, Batch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import spacy
import clip
from tqdm import tqdm

import networkx as nx
from itertools import combinations
import numpy as np

from datasets import load_dataset

import matplotlib.pyplot as plt
import re

import pickle
import random

class MakeGraphKai5Etype59plus:
    def __init__(self, use_bert_base = True):
        # spaCy
        self.nlp = spacy.load("en_core_web_sm")
        #self.nlp = spacy.load("en_core_web_trf")

        self.deprel_59 = [
            "nsubj", "obj", "iobj", # Nominals - Core arguments
            "csubj", "ccomp", "xcomp", # Clauses - Core arguments
            "obl", "vocative", "expl", "dislocated", # Nominals - Non-core dependents
            "advcl", # Clauses - Non-core dependents
            "advmod", "discourse", # Modifier words - Non-core dependents
            "aux", "cop", "mark", # Function Words - Non-core dependents
            "nmod", "appos", "nummod", # Nominals - Nominal dependents
            "acl", # Clauses - Nominal dependents
            "amod", # Modifier words - Nominal dependents
            "det", "clf", "case", # Function Words - Nominal dependents
            "conj", "cc", # Coordination
            "fixed", "flat", # Headless
            "list", "parataxis", # Loose
            "compound", "orphan", "goeswith", "reparandum", #Special
            "punct", "ROOT", "dep", # Other

            # not in UD x2, maybe
            "acomp", 
            "preconj",
            "npadvmod",
            "oprd",
            "agent",
            "relcl", 
            "nsubjpass", 
            "dobj", 
            "pobj", 
            "dative", 
            "quantmod", 
            "prt", 
            "pcomp",
            "prep", 
            "predet",
            "meta", 
            "csubjpass",
            "intj", 
            "neg", 
            "poss", 
            "auxpass", 
            "attr"
        ]
        self.deprel_counter = [0 for i in range(59)]
        self.out_dep_set = set()
        self.in_dep_g = 0
        self.out_dep_g = 0
        
        # BERT
        if use_bert_base:
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        else:
            print("use bert-large-uncased!")
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            self.bert_model = BertModel.from_pretrained("bert-large-uncased")
        # self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # self.bert_model = RobertaModel.from_pretrained('roberta-base')

        self.tokenized_by_spacy = []#spaCyがトークナイズした結果を格納
        self.tokenized_by_spacy_sen1 = []
        self.tokenized_by_spacy_sen2 = []
        self.node_ft_error = False

        self.miss_graph_cnt = 0
        self.data_img_id = 0

        self.sumple_img_flag = 5
        
    def trans_dataset_kai_type2(self, datas, header, save_name):# list in dict (ex. by load_datasets_mod2)
        self.in_dep_g = 0
        self.out_dep_g = 0
        self.miss_graph_cnt = 0
        self.data_img_id = 0
        
        print(f"start make graph of {save_name}")
        # header = [sentence1, sentence2, label]
        concated_sentences = []
        cleanuped_sentence1s = []

        sentence1_l = []
        sentence2_l = []
        labels_l = []
        for data in datas:
            concated_sentences.append(self.__cleanup_sentence(data[header[0]]) + " " + self.__cleanup_sentence(data[header[1]]))
            cleanuped_sentence1s.append(self.__cleanup_sentence(data[header[0]]))

        gdatas, deficit_list = self.__get_graph(concated_sentences, cleanuped_sentence1s, save_name)

        print("len(gdatas):", len(gdatas))
        tmp_counter = 0
        for i, data in enumerate(datas):
            if deficit_list[i]:# グラフの生成に成功した文のみ追加
                sentence1_l.append(data[header[0]])
                sentence2_l.append(data[header[1]])
                labels_l.append(data[header[2]])
                tmp_counter += 1
        print("tmp_counter:", tmp_counter)
        removed_sentence_num = len(datas) - tmp_counter
        print(f"removed {removed_sentence_num} sentences")
        print(f"success rate {tmp_counter / len(datas)}")

        # CLIP用に文を分解
        #mini_sentences1 = self.__disassemble_sentence(sentence1_l)
        #mini_sentences2 = self.__disassemble_sentence(sentence2_l)
        
        # データを再格納
        datas_add_graph = {}
        #データセット間での違いをなくすため，"sentence1"等に固定 20240904
        datas_add_graph["sentence1"] = sentence1_l
        datas_add_graph["sentence2"] = sentence2_l
        datas_add_graph["label"] = labels_l

        datas_add_graph["graphdata"] = gdatas
        #datas_add_graph["mini_sentences1"] = mini_sentences1
        #datas_add_graph["mini_sentences2"] = mini_sentences2
        #print("datas_add_graph", datas_add_graph)

        print(f"not in self.deprel_59 is\n {self.out_dep_set}")
        print(f"self.in_dep_g: {self.in_dep_g}")
        print(f"self.out_dep_g: {self.out_dep_g}")

        # 保存
        with open(save_name + ".pickle", mode='wb') as fo:
            pickle.dump(datas_add_graph, fo)
        print("SAVED:", save_name)
    
        print(f"self.miss_graph_cnt: {self.miss_graph_cnt}")

        print("\nself.deprel_counter:")
        for i, deprel_count in enumerate(self.deprel_counter):
            print(f"{i}:{self.deprel_59[i]}:{deprel_count}")
        print(f"{len(self.deprel_59)}:plus edge")

        self.sumple_img_flag = 5
        return datas_add_graph
    
    def __cleanup_sentence(self, sentence):# 小文字化＆省略形の排除など
        tmp_sentence = sentence.lower()# 小文字化
        tmp_sentence = tmp_sentence.replace("  ", " ")
        tmp_sentence = tmp_sentence.replace("\xad", "") # ソフトハイフンの除去
        tmp_sentence = tmp_sentence.replace("\xa0", "") # ノーブレークスペースの除去

        # 省略形の修正
        tmp_sentence = tmp_sentence.replace("won't", "will not")
        tmp_sentence = tmp_sentence.replace("wont", "will not")
        tmp_sentence = tmp_sentence.replace("willnt", "will not")
        tmp_sentence = tmp_sentence.replace("willn 't", "will not")
        tmp_sentence = tmp_sentence.replace("'ll", " will")

        tmp_sentence = tmp_sentence.replace("wouldnt", "would not")
        tmp_sentence = tmp_sentence.replace("wouldn 't", "would not")

        tmp_sentence = tmp_sentence.replace("canot", "can not")
        tmp_sentence = tmp_sentence.replace("cant", "can not")
        tmp_sentence = tmp_sentence.replace("cannt", "can not")
        tmp_sentence = tmp_sentence.replace("cannot", "can not")
        tmp_sentence = tmp_sentence.replace("can't", "can not")
        tmp_sentence = tmp_sentence.replace("can 't", "can not")

        tmp_sentence = tmp_sentence.replace("couldnt", "could not")
        tmp_sentence = tmp_sentence.replace("couldn 't", "could not")

        tmp_sentence = tmp_sentence.replace("maynt", "may not")
        tmp_sentence = tmp_sentence.replace("mayn 't", "may not")

        tmp_sentence = tmp_sentence.replace("mightnt", "might not")
        tmp_sentence = tmp_sentence.replace("mightn 't", "might not")

        tmp_sentence = tmp_sentence.replace("mustnt", "must not")
        tmp_sentence = tmp_sentence.replace("mustn 't", "must not")

        tmp_sentence = tmp_sentence.replace("shouldnt", "should not")
        tmp_sentence = tmp_sentence.replace("shouldn 't", "should not")

        tmp_sentence = tmp_sentence.replace("neednt", "need not")
        tmp_sentence = tmp_sentence.replace("needn 't", "need not")

        tmp_sentence = tmp_sentence.replace("havent", "have not")
        tmp_sentence = tmp_sentence.replace("haven 't", "have not")

        tmp_sentence = tmp_sentence.replace("dont", "do not")
        tmp_sentence = tmp_sentence.replace("don 't", "do not")

        tmp_sentence = tmp_sentence.replace("doesnt", "does not")
        tmp_sentence = tmp_sentence.replace("doesn 't", "does not")
        
        tmp_sentence = tmp_sentence.replace("didn 't", "did not")
        tmp_sentence = tmp_sentence.replace("didnt", "did not")

        tmp_sentence = tmp_sentence.replace("'re", "are")
        tmp_sentence = tmp_sentence.replace("'m", "am")
        tmp_sentence = tmp_sentence.replace("'ve", "have")

        tmp_sentence = tmp_sentence.replace("n't", " not")
        tmp_sentence = tmp_sentence.replace(" nt ", " not ")# kai4-2（2025/01/15の朝以降のモデル）
        tmp_sentence = tmp_sentence.replace("  ", " ")# 再度スペースが2回続く箇所を削除

        # Latexの数式の除去
        tmp_sentence = re.sub(r"\[math\].*?\[\/math\]", "this formula", tmp_sentence)

        if tmp_sentence[-1] != "." and tmp_sentence[-1] != "?":
            # if self.sumple_img_flag > 0:
            #     print(f"add '.' : {tmp_sentence}")
            tmp_sentence += " ."
            # if self.sumple_img_flag > 0:
            #     print(f"after : {tmp_sentence}")
        return tmp_sentence
        
    def __get_graph(self, sentences, sentence1s, save_name):
        sample_print = True
        gdatas = []
        deficit_list = []
        cleanuped_sentences = [s.replace("  ", " ") for s in sentences]
        
        # spaCyで処理
        docs = list(self.nlp.pipe(cleanuped_sentences))
        doc1s = list(self.nlp.pipe(sentence1s))# sentence1のみをspaCyに処理させる．トークン数カウント用

        #for sentence, doc in tqdm(zip(cleanuped_sentences, docs)):
        for sentence, doc, doc1 in zip(cleanuped_sentences, docs, doc1s):
            self.tokenized_by_spacy = []
            self.tokenized_by_spacy_sen1 = []
            self.tokenized_by_spacy_sen2 = []

            edges = []
            tmp_edge_type = []
            tmp_tokenized = {}
            for ti, token in enumerate(doc):
                for child in token.children:# spaCyから得た係り受け関係を元にエッジを追加
                    # if child.dep_ in self.popular_deprel:# 独自に設定した主要な係り受け関係であれば
                    #     edges.append((token.i, child.i))
                    #     tmp_edge_type.append(self.popular_deprel.index(child.dep_))

                    dep_num =  self.__get_dep_num(child.dep_)
                    if dep_num != 9999:# https://universaldependencies.org/u/dep/　を元にした係り受け関係の表を参照
                        edges.append((token.i, child.i))
                        tmp_edge_type.append(dep_num)

                # 共通単語のエッジを追加
                if ti >= len(doc1) and token.text in list(tmp_tokenized.keys()): # 既に出現済みの単語の場合，共起と判断しエッジを追加
                    for exited_word_i in tmp_tokenized[token.text]:
                        edges.append((exited_word_i, token.i))
                        tmp_edge_type.append(len(self.deprel_59))# 係り受け関係の総数+1を「共起」の関係に設定

                        edges.append((token.i, exited_word_i))#双方向とするため，入れ替えたものも追加？
                        tmp_edge_type.append(len(self.deprel_59))# 係り受け関係の総数+1を「共起」の関係に設定

                elif ti < len(doc1):#文内では繋がないよう，既出のエッジの追加は1文目のみで行う
                    if  token.text in list(tmp_tokenized.keys()):
                        tmp_tokenized[token.text].append(token.i)
                    else:
                        tmp_tokenized[token.text] = [token.i]
                

                self.tokenized_by_spacy.append(token.text)
                
                if ti < len(doc1):#1文目のトークナイズ結果を保存
                    self.tokenized_by_spacy_sen1.append(token.text)
                else:
                    self.tokenized_by_spacy_sen2.append(token.text)

            if len(edges) == 0:# エッジが存在しなければ
                #deficit_list.append(False)
                #self.miss_graph_cnt += 1
                print("len(edge_index) == 0")
                print("edge_index is NONE!!\nCant't make graph :", sentence)


            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(tmp_edge_type, dtype=torch.long)
            # print("sentence:", sentence)
            # print("edge_index:", edge_index)
            # print("edge_type:", edge_type)

            node_fts = self.__get_ft()#[SEP]を文の間に挿入させるため，2文を渡す
            #node_fts = torch.tensor([token.vector for token in doc], dtype=torch.float)
            #print(Data(x=node_fts, edge_index=edge_index, edge_type=edge_type))
            if self.node_ft_error:
                self.node_ft_error = False
                deficit_list.append(False)
                self.miss_graph_cnt += 1
                print("node_ft_error == True")
                print("Cant't make graph :", sentence)
            else:
                # エッジの指し示すノードが存在するのか確認
                # for edg_i in edge_index:
                #     print("edg_i:", edg_i)
                #     print("len(node_fts):", len(node_fts))
                #gdatas.append(Data(x=node_fts, edge_index=edge_index, edge_type=edge_type))
                gdatas.append({"x":node_fts, "edge_index":edge_index, "edge_type":edge_type, "num_nodes":len(node_fts), "num_of_nodes_sentence1":len(doc1)})
                deficit_list.append(True)
                if self.sumple_img_flag > 0:
                    self.sumple_img_flag -= 1
                    self.__plot_sentence_graph(sentence, self.tokenized_by_spacy, edge_index, len(doc1), save_name)# グラフを画像で保存

                if sample_print:
                    sample_print = False
                    print("sample of gdatas")
                    print(f"sentence     : {sentence}")
                    print(f"sentences[0] : {sentences[0]}")
                    print(f"sentence1s[0]: {sentence1s[0]}")
                    print(f"gdata[0]     : {gdatas[0]}")
                
            #gdatas.append({"x":node_fts, "edge_index":edge_index, "edge_type":edge_type})
        return gdatas, deficit_list
        
    def __get_ft(self):
        bert_tokens = self.bert_tokenizer(self.tokenized_by_spacy_sen1, self.tokenized_by_spacy_sen2, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = bert_tokens["input_ids"][0]
        tokens = self.bert_tokenizer.convert_ids_to_tokens(input_ids)
        # print(f"Tokens: {tokens}")
        # print(f"Sequence Length (with special tokens): {len(tokens)}")

        # Get BERT token-level embeddings
        with torch.no_grad():  # Disable gradient calculation as we only need the output
            outputs = self.bert_model(**bert_tokens)
            # The last hidden state is the token-level embedding
            token_embeddings = outputs.last_hidden_state
        
        pooled_embeddings = []
        pooled_words = []
        last_embs = []
        last_words = []
        
        token_cursor = 0 # その位置のspaCyのトークンに対応するBERT埋め込み表現をpool
        existed_first_sep = False
        for b_token, emb in zip(tokens, token_embeddings[0]):
            #print(b_token)
            if b_token in ["[CLS]", "[SEP]"]:
                if b_token == "[SEP]" and existed_first_sep == True:
                    pooled_embeddings.append((sum(last_embs)/len(last_embs)).clone().detach())
                    pooled_words.append(last_words)
                    existed_first_sep = False
                    #print("existed_first_sep", existed_first_sep)
                    #token_cursor += 1
                elif b_token == "[SEP]":
                    existed_first_sep = True
                    #print("existed_first_sep", existed_first_sep)
                continue
            
            if b_token.startswith("##"):
                #単語の続きなので追加格納
                last_embs.append(emb)
                last_words.append(b_token)
            else:
                # 現時点でのlast_wordsを連結
                last_words_pooled = ""
                for lw in last_words:
                    last_words_pooled += lw.replace("##", "")

                #単語の続きであると判断
                #if last_words_pooled != self.tokenized_by_spacy[token_cursor]:
                if len(self.tokenized_by_spacy) < (token_cursor + 1):
                    token_cursor -= 1 
                if len(last_words_pooled) < len(self.tokenized_by_spacy[token_cursor]) and self.tokenized_by_spacy[token_cursor].startswith(last_words_pooled):
                    last_embs.append(emb)
                    last_words.append(b_token)

                elif last_words_pooled.startswith("[UNK]") or last_words_pooled.endswith("[UNK]"):# token_cursorの指す単語が非アルファベットの可能性大
                    if (len(self.tokenized_by_spacy) - 1) >= (token_cursor + 1):
                        if self.tokenized_by_spacy[token_cursor + 1].startswith(b_token):# 次の単語がb_tokenで始まっているのなら続いていない
                            if len(last_embs) != 0:
                                pooled_embeddings.append((sum(last_embs)/len(last_embs)).clone().detach())
                                pooled_words.append(last_words)
                                token_cursor += 1
                            #初期化して格納
                            last_embs = []
                            last_embs.append(emb)
                            last_words = []
                            last_words.append(b_token)
                    else:# 未知語が続いている?
                        last_embs.append(emb)
                        last_words.append(b_token)

                else:# 続いてないので前の埋め込みをpoolして追加
                    if len(last_embs) != 0:
                        #pooled_embeddings.append(torch.tensor(sum(last_embs)/len(last_embs)))
                        pooled_embeddings.append((sum(last_embs)/len(last_embs)).clone().detach())
                        pooled_words.append(last_words)
                        token_cursor += 1
                
                    #初期化して格納
                    last_embs = []
                    last_embs.append(emb)
                    last_words = []
                    last_words.append(b_token)

        if len(self.tokenized_by_spacy) != len(pooled_embeddings):# spaCyのトークン数とpool後のノード数が合わない場合
            self.node_ft_error = True

            print("==node_ft_error==")
            print(f"self.tokenized_by_spacy: {self.tokenized_by_spacy}")
            print(f"pooled_words: {pooled_words}")
            print(f"self.tokenized_by_spacy length: {len(self.tokenized_by_spacy)}")
            print(f"pooled_embeddings length: {len(pooled_embeddings)}")
        else:
            self.node_ft_error = False

        return (torch.stack(pooled_embeddings)).clone().detach()
    
    def __get_dep_num(self, child_dep):
        for i, g in enumerate(self.deprel_59):
            if child_dep == g:
                self.in_dep_g += 1
                self.deprel_counter[i] += 1
                return i

        # print(f"{child_dep} is not in self.deprel_59")
        self.out_dep_g += 1
        self.out_dep_set.add(child_dep)
        return 9999

    def __plot_sentence_graph(self, concated_sentences, tokenized_by_s, edge_index, N, save_dir):
        # グラフを作成
        G = nx.DiGraph()  # 係り受け関係などを表すため有向グラフにする

        # ノードを追加（前半と後半の文の単語に応じて色分け）
        # print(f"concated_sentences\n{concated_sentences}")
        # print(f"tokenized_by_s\n{tokenized_by_s}")
        # print(f"edge_index\n{edge_index}")
        # print(f"N: {N}")
        # print(f"save_dir: {save_dir}")

        for i, word in enumerate(tokenized_by_s):
            if i < N:
                G.add_node(i, label=word, color="blue")  # sentence1の単語は青色
            else:
                G.add_node(i, label=word, color="orange")  # sentence2の単語はオレンジ色

        # エッジを追加 (tensorをリストに変換して扱う)
        edge_index_list = edge_index.tolist()
        for src, dst in zip(edge_index_list[0], edge_index_list[1]):
            G.add_edge(src, dst)

        node_colors = [G.nodes[i]["color"] for i in G.nodes]# ノードの色を取得        
        labels = {i: G.nodes[i]["label"] for i in G.nodes}# ノードのラベルを取得

        # グラフの描画
        #pos = nx.spring_layout(G)  # ノードの配置を決定
        #plt.figure(figsize=(10, 7))
        
        # カスタムレイアウト: ノードを登場順に等間隔で配置
        pos = {}
        x_interval = 5 / len(tokenized_by_s)  # ノード間のx座標の間隔
        for i in G.nodes:
            x_coord = i * x_interval - 1  # -1 から 1 の範囲に x 座標を設定
            if i < N:
                pos[i] = (x_coord, random.uniform(0.5, 1))  # 青色ノードは上部に配置
            else:
                pos[i] = (x_coord, random.uniform(-1, -0.5))  # オレンジ色ノードは下部に配置

        plt.figure(figsize=(12, 9))  # より大きなサイズにする

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)# ノードを描画
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=15)# エッジを描画
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='white')# ラベルを描画

        # グラフの保存
        # concated_sentencesをファイル名にするため、使用できない文字を削除または置き換え
        # max_filename_length = 20  # 最大のファイル名の長さを設定
        # if len(concated_sentences) > max_filename_length:
        #     short_filename = concated_sentences[:25] + '...' + concated_sentences[-20:]
        # else:
        #     short_filename = concated_sentences
        # safe_filename = re.sub(r'[^\w\-_\(\)]', '_', short_filename)
        safe_filename = save_dir + "_" + str(self.data_img_id)
        plt.title(concated_sentences.replace("$", ""))
        plt.axis('off')
        plt.savefig(f"./gdata_images_kai5/{save_dir}/{safe_filename}.png")  # 画像を保存
        plt.close()
        self.data_img_id += 1

    def len_deprel(self):# RGCNのnum_relations用
        num_of_deprel_g = len(self.deprel_59) + 1
        print(f"num_of_deprel_g:{num_of_deprel_g}")
        return num_of_deprel_g
    
    # 文を分解
    def __disassemble_sentence(self, sentences):
        mini_sentences = []
        for sentence in sentences:
            tmp_mini = []
            doc = self.nlp(sentence)
            subgraphs = self.__create_subgraphs(doc)# サブグラフを取得

            # 各サブグラフのテキストを表示
            for i, sg in enumerate(subgraphs):
                #print(f"Subgraph {i+1}:")
                nodes = sorted(sg.nodes())
                subgraph_text = ' '.join([doc[node].text for node in nodes])
                # if subgraph_text.count(" ") < 70:# 意味なし？
                if "\\" in subgraph_text:
                    #print("REMOVE:", subgraph_text)
                    pass
                else:
                    tmp_mini.append(subgraph_text)
                # else:
                #     print("remove subgraph_text")
                #     print(f"=>{subgraph_text}")
            mini_sentences.append(tmp_mini)
        # for ms in mini_sentences:
        #     print(ms)
        if len(mini_sentences) < 1:
            print("mini_sentences is empty!")
        return mini_sentences
    
    def debag_disassemble_sentence(self, sentences):
        mini_sentences = self.__disassemble_sentence(sentences)
        print("mini_sentences")
        for s, mini_s in zip(sentences, mini_sentences):
            print(f"org:{s}")
            print(f"{mini_s}\n")

    # 依存関係ラベルごとにサブグラフを作成する関数
    def __create_subgraphs(self, doc):# self.nlp()から得たdocを渡す
        subgraphs = []
        for token in doc:
            children_count = 0
            subgraph = nx.Graph()
            #print(token.pos_)
            if token.pos_ in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]:#, "ADP"]:#,  "AUX", "DET", "NUM", "PRON"]:
                subgraph.add_node(token.i, label=token.text)
                for child in token.children:
                    children_count += 1
                    subgraph.add_node(child.i, label=child.text)
                    subgraph.add_edge(token.i, child.i, label=child.dep_)
                if children_count > 0:
                    subgraphs.append(subgraph)
        return subgraphs
    

class MyTextGraphDatasetKai5(Dataset):
    def __init__(self, data):
        self.data = []
        # print("len(data[sentence1])", len(data["sentence1"]))
        # print("len(data[sentence2])", len(data["sentence2"]))
        # print("len(data[label])", len(data["label"]))
        # print("len(data[graphdata])", len(data['graphdata']))
        # print("len(data[mini_sentences1])", len(data['mini_sentences1']))
        # print("len(data[mini_sentences2])", len(data['mini_sentences2']))

        # データ長の不一致をチェック
        lengths = [len(data[key]) for key in ["sentence1", "sentence2", "label", "graphdata"]]
        if len(set(lengths)) != 1:
            raise ValueError("All input lists must have the same length")

        for sentence1, sentence2, label, gdata in zip(data["sentence1"], data["sentence2"], data["label"], data['graphdata']):
            tmp_data = {}
            tmp_data["sentence1"] = sentence1
            tmp_data["sentence2"] = sentence2
            tmp_data["label"] = label

            tmp_data["graphdata"], tmp_data["num_of_nodes_sentence1"] = self.__convert_graph_data(gdata)
            #tmp_data["mini_sentences1"] = self.__consolidate_mini_sentences(mini_s1)
            #tmp_data["mini_sentences2"] = self.__consolidate_mini_sentences(mini_s2)
            
            self.data.append(tmp_data)

    def __consolidate_mini_sentences(self, mini_sentences):
        #return "<split>".join(mini_sentences)
        consolidate_text = ""
        for i, ms in enumerate(mini_sentences):
            consolidate_text += ms
            if (i+1) < len(mini_sentences):
                consolidate_text += "<split>"
        #print("consolidate_text", consolidate_text)
        return consolidate_text

    def __convert_graph_data(self, gdata):
        # gdataが期待される形式であることを確認する
        try:
            return Data(x=gdata["x"], edge_index=gdata["edge_index"], edge_attr=gdata["edge_type"], num_nodes=gdata["num_nodes"]), gdata["num_of_nodes_sentence1"]
        except AttributeError as e:
            raise ValueError(f"Invalid graph data format: missing attribute {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            print(f"Index {idx} out of range")
            raise IndexError(f"Index {idx} out of range")
        return self.data[idx]


if __name__ == "__main__":
    from load_datasets_kai_mod0_fixPIT2015dev import QQP, PAWS_QQP, PAWS_Wiki, PIT2015, MRPC

    # データの準備
    dataset_names = ["qqp", "pawsqqp", "pawswiki", "pit2015", "mrpc"]
    dataset_name = dataset_names[3]

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
    
    dataset = {}
    dataset["train"] = dataset_source.get_train_data()
    dataset["validation"] = dataset_source.get_dev_data()
    dataset["test"] = dataset_source.get_test_data()
    print(f"loaded dataset {dataset_name}")
    header = dataset_source.get_header()
    

    # グラフデータの生成と追加
    use_bert_base_flag = True
    graph_data_maker = MakeGraphKai5Etype59plus(use_bert_base = use_bert_base_flag)
    graph_data_maker.len_deprel()
    
    if use_bert_base_flag:
        L_flag = ""
    else:
        L_flag = "L-"
    
    #dataset_train = graph_data_maker.trans_dataset_kai_type2(dataset["train"], header, f"{dataset_name}-Etype59plus-kai5-{L_flag}train")
    dataset_validation = graph_data_maker.trans_dataset_kai_type2(dataset["validation"], header, f"{dataset_name}-Etype59plus-kai5-{L_flag}validation_Guidelined")
    #dataset_test = graph_data_maker.trans_dataset_kai_type2(dataset["test"], header, f"{dataset_name}-Etype59plus-kai5-{L_flag}test")

    print("FINISH!!!")
