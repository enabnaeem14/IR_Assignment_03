import pandas as pd
import numpy as np
from search import IRSystem
from collections import defaultdict
import time

def precision_at_k(ret, rel, k):
    return sum(1 for d in ret[:k] if d in rel) / k

def mean_avg_precision(rets, rels, k=10):
    APs = []
    for qid, ret in rets.items():
        rel = rels.get(qid, set())
        if not rel: continue
        hits = 0
        sum_prec = 0
        for i, d in enumerate(ret[:k], start=1):
            if d in rel:
                hits += 1
                sum_prec += hits / i
        APs.append(sum_prec / len(rel))
    return np.mean(APs)

if __name__ == "__main__":
    qrels = pd.read_csv("qrels.csv")
    queries = qrels.groupby("qid").first()["query"].to_dict()
    rels = qrels[qrels["relevance"] == 1].groupby("qid")["doc_id"].apply(set).to_dict()

    ir = IRSystem()

    retrieved = {}
    timings = []

    for qid, q in queries.items():
        t0 = time.time()
        idx, _ = ir.hybrid(q, k=100)
        t1 = time.time()
        retrieved[qid] = idx.tolist()
        timings.append(t1 - t0)

    print("MAP@10 =", mean_avg_precision(retrieved, rels, k=10))
    print("Precision@5 =", np.mean([precision_at_k(retrieved[q], rels.get(q, set()), 5) for q in queries]))
    print("Mean Latency =", np.mean(timings))
