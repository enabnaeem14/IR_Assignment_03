# CS516 – Information Retrieval Project 
# IR Project – News Articles IR System

This project implements a complete **local Information Retrieval (IR) system** on the Kaggle *News Articles* dataset.  
The system uses **TF–IDF**, **BM25**, and a **hybrid ranking model**, and includes **manual qrels-based evaluation**.

Everything runs **locally** with no cloud services. 

## Overview
A fully local Information Retrieval (IR) system using:
- Text preprocessing (tokenization, stopwords, lemmatization)
- TF–IDF Vector Space Model
- BM25 Ranking
- Hybrid Ranking
- Manual Qrels Evaluation (MAP, Precision@k, Recall)

Dataset: Kaggle "News Articles"  
Link: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles

---

## Installation
 
## Environment Setup
Step — Create Conda Environment
```bash
conda create -n new_env
conda activate new_env

## Install Requirements
```bash
pip install -r requirements.txt

## DataSet Setup
1. Download `news_articles.csv` from the Kaggle dataset: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles using download_data.py and place it in `data/`. 
    python src/download_data.py
## NLTK Setup (Required Once)
2. Prepare NLTK resources (run once): 
     python -m nltk.downloader punkt wordnet stopwords
## Build INdex     
3. Build index:
     python src/indexer.py --data_path data/news_articles.csv --out_dir indexes/
this creates
   indices/
   ├── tfidf.joblib
   ├── bm25.joblib
   └── docs.parquet


##  Run Search Queries    
4. Run simple query: 
     python src/search.py --query "election campaign rally" --k 10
     this will show top -10 ranked results
## Qrels (Relevance Judgements)     
5. Evaluate (requires relevance judgments in `data/qrels.csv`):
format of qrel.csv:
            qid,query,doc_id,relevance
             1,political news,120,1
             1,political news,45,1
             1,political news,789,0
 Where:
doc_id = row index of the document
relevance:
1 → relevant
0 → not relevant
At least 10 judgments per query were created manually.    

python src/evaluate.py --data/qrels.csv
These were calculated : Outputs:

MAP@10
Precision@5
Recall@10
Mean query latency

These were used in the final report.

# Project Structure 
ASSIGNMENT_03/
│
├── data/
│   ├── news_articles.csv
│   └── qrels.csv
│
├── indices/
│   ├── tfidf.joblib
│   ├── bm25.joblib
│   └── docs.parquet
│
├── src/
│   ├── preprocess.py # tokenization, stopwords, lemmatize, clean
│   ├── indexer.py # build TF-IDF and BM25 indexes, save to disk
│   ├── search.py # IRSystem Class,query interface: boolean, BM25, TF-IDF cosine
│   ├── evaluate.py # load qrels and compute metrics: P@k, MAP, latency
│   └── download_data.py #  Kaggle data downloader
│
├── queries_results.txt
├── requirements.txt
└── README.md


## Reproducibility
- All steps run locally; no cloud services required.
- Evaluation uses manually created qrels
- Code tested on Python 3.10

## How to Fully Reproduce
pip install -r requirements.txt
python -m nltk.downloader punkt wordnet stopwords
python src/download_data.py
python src/indexer.py --data data/news_articles.csv --outdir indices/
python src/search.py --query "test query" --k 5
python src/evaluate.py



## Author 
- Enab Naeem
- MSDS24014
- Submited as Assignment_03