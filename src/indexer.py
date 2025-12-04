import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from src.preprocess import tokenize_and_lemmatize, collapse_to_string
from tqdm import tqdm

def load_data(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")  # WINDOWS ENCODING

    df = df.fillna('')
    df["text"] = df["Heading"].astype(str) + ". " + df["Article"].astype(str)
    return df

def preprocess_texts(df):
    tokenized = []
    collapsed = []

    for t in tqdm(df["text"], desc="Preprocessing"):
        toks = tokenize_and_lemmatize(t)
        tokenized.append(toks)
        collapsed.append(collapse_to_string(toks))

    df["tokens"] = tokenized
    df["tokens_str"] = collapsed
    return df

def build_tfidf_index(texts, save_path):
    vect = TfidfVectorizer(max_features=100000, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    joblib.dump((vect, X), save_path)
    print("TF-IDF saved:", save_path)
    return vect, X

def build_bm25_index(tokenized_texts, save_path):
    bm25 = BM25Okapi(tokenized_texts)
    joblib.dump(bm25, save_path)
    print("BM25 saved:", save_path)
    return bm25

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="indices")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.data)
    df = preprocess_texts(df)

    build_tfidf_index(df["tokens_str"].tolist(), f"{args.outdir}/tfidf.joblib")
    build_bm25_index(df["tokens"].tolist(), f"{args.outdir}/bm25.joblib")

    df.to_parquet(f"{args.outdir}/docs.parquet", index=False)
    print("Indexing complete.")
