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


### other
coming soon
