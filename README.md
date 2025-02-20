# Integration of sentence-embedding-generation LLM and RGCN for paraphrase identification

## Datasets
Datasets used in this study are available at the following links (accessed on 13 February 2025)
- QQP : https://github.com/zhiguowang/BiMPM/tree/master?tab=readme-ov-file
- PIT2015 : https://github.com/cocoxu/SemEval-PIT2015/tree/master
- MRPC : https://huggingface.co/datasets/nyu-mll/glue

## Usage
### create conda environment
```
conda update -n base conda
conda create -n cuda_pytorch_gnn_clip_1 python=3.10.12
python -m spacy download en_core_web_sm

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install conda-forge::transformers

conda install -c conda-forge sentence-transformers

conda install pyg -c pyg

conda install anaconda::networkx
conda install matplotlib

conda install ftfy regex tqdm
```


### Usage
1. Make Graph
Create graph data that expresses the dependency relationships of sentences.
Given that we will be processing a large amount of data, this time we will save graph data in pickle format and use this for model training, etc.
Please run the following command
```
python make_graph_Etype59plus.py
```
or 
```
python make_graph_Etype59.py
```
Etype59 does not have edges connecting sentence pairs.
Dataset selection is performed by the index of ```dataset_names[]``` assigned to ```dataset_name```.
Please move the generated pickle file to the directory "pickle_datas".

2. Training
We are preparing code to train three types of models. Select any one and execute it.
For example, if you want to train CombinedModel, run the following command
```
python train_combine_rgcn_ly3_PromptEOL_asub_mul.py
```

3. Test
We are preparing code to test three types of models. Select any one and execute it.
For example, if you want to test CombinedModel, run the following command
```
python test_combine_rgcn_ly3_PromptEOL_asub_mul.py
```
The epoch size etc. will be displayed during the test, but please ignore them.
Details on how to specify datasets and models will be described later.