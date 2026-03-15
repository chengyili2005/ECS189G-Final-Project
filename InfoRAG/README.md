# InfoRAG
Replicating findings from [Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation](https://arxiv.org/pdf/2402.18150) on smaller parameter models.

Setup: (In your environment of choice)
1. Download packages
```{bash}
pip install -r requirements.txt
```

2. Download Wikipedia articles
```{bash}
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```
3. Run script
```{bash}
python main.py
```