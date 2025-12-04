import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import tokenize_and_lemmatize, collapse_to_string
import pandas as pd

class IRSystem:
    def __init__(self, indices_dir="indices"):
        self.vect, self.tfidf_matrix = joblib.load(f"{indices_dir}/tfidf.joblib")
        self.bm25 = joblib.load(f"{indices_dir}/bm25.joblib")
        self.df = pd.read_parquet(f"{indices_dir}/docs.parquet")

       
    def query_tfidf(self, query, k=10):
        qtok = collapse_to_string(tokenize_and_lemmatize(query))
        qvec = self.vect.transform([qtok])
        sims = cosine_similarity(self.tfidf_matrix, qvec).ravel()
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]

    def query_bm25(self, query, k=10):
        q_tokens = tokenize_and_lemmatize(query)
        scores = self.bm25.get_scores(q_tokens)
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]

    def hybrid(self, query, k=10, alpha=0.5):
        idx_b, s_b = self.query_bm25(query, k=len(self.df))
        idx_t, s_t = self.query_tfidf(query, k=len(self.df))

        bm = np.zeros(len(self.df))
        tf = np.zeros(len(self.df))
        bm[idx_b] = s_b
        tf[idx_t] = s_t

        final = alpha * bm + (1 - alpha) * tf
        top = np.argsort(-final)[:k]
        return top, final[top]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    ir = IRSystem()
    idx, scores = ir.hybrid(args.query, k=args.k)
    
    print("\n=== Search Results ===")
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        print(f"rank={rank} | doc_id={i} | score={s:.4f} | title={ir.df.iloc[i]['Heading'][:150]}")

