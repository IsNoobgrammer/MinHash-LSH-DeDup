from minhash import embed_func,INDEX_COLUMN,NON_ALPHA,SIGNATURE_COLUMN
from hashs import sha1_hash
from utils import optimal_param
from union_find import UnionFind


import numpy as np
from collections import defaultdict
from datasets import load_dataset,Dataset
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import random

if os.name != "nt":
    mp.set_start_method("fork", force=True)


def deduplicate_dataset(
    ds: Dataset,
    column,
    threshold=0.8,
    num_perm=256,
    batch_size=10_000,
    num_proc= 1 if os.name == "nt" else os.cpu_count() ,
    ngram_size=5,
    min_length=5,
    bands_rows=(None,None)
    ) -> Dataset :



    SEED = 42
    RNG = np.random.RandomState(SEED)
    BITS=64
    DTYPE, MAX_HASH, MODULO_PRIME = np.uint64,np.uint32((1 << 32) - 1),np.uint64((1 << 61) - 1)
    hash_func=lambda x: sha1_hash(x, d=BITS)

    
    
    CLUSTER_COLUMN="__cluster__"
    uf = UnionFind()
    BANDS,ROWS=bands_rows
    if BANDS == None or ROWS==None:
        BANDS,ROWS=optimal_param(
                    threshold=threshold,
                    num_perm=num_perm,
                    false_positive_weight=0.5,
                    false_negative_weight=0.5,
                )
    
    HASH_RANGES = [(i * ROWS, (i + 1) * ROWS) for i in range(BANDS)]
    HASH_TABLES: list[dict[int, set]] = [defaultdict(set) for _ in range(BANDS)]
    PERMUTATIONS: tuple[np.ndarray, np.ndarray] = (
            RNG.randint(
                1, MODULO_PRIME, size=(num_perm,), dtype=DTYPE
            ),  # a is a multiplier so should not be 0
            RNG.randint(0, MODULO_PRIME, size=(num_perm,), dtype=DTYPE),  
    )

         
    ds = ds.map(lambda x, i: {INDEX_COLUMN: i}, with_indices=True, num_proc=num_proc,desc="Indexing")
    ds = ds.filter(
                    lambda x: len(NON_ALPHA.split(str(x[column]).lower())) >= min_length , 
                    num_proc=num_proc,
                )
    
    LEN_DATASET = len(ds)

    ### MinHashing
    embedded = ds.map(
                    function=embed_func,
                    fn_kwargs={
                        "num_perm": num_perm,
                        "hashranges": HASH_RANGES,
                        "ngram_size": ngram_size,
                        "min_length": min_length,
                        "permutations": PERMUTATIONS,
                        "hash_func": hash_func,
                        "dtype": DTYPE,
                        "max_hash": MAX_HASH,
                        "modulo_prime": MODULO_PRIME,
                    },
                    input_columns=[column, INDEX_COLUMN],
                    remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                    num_proc=num_proc,
                    with_indices=False,
                    batched=True,
                    batch_size=batch_size,
                    desc="Min-Hashing...",
                )

    # print(embedded[0])
    LEN_EMBEDDED = len(embedded)
    NUM_SHARDS = np.ceil(LEN_EMBEDDED / batch_size).astype(int)
    edges = []
    for i in tqdm(
        range(0, NUM_SHARDS),
        dynamic_ncols=True,
        desc="Iterating MinHashes...",  # noqa: E501
    ):
        embedded_shard = embedded.shard(
            num_shards=NUM_SHARDS,
            index=i,
            contiguous=True,
            writer_batch_size=batch_size,
        )
        for key, Hs in zip(embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]):
            for i, H in enumerate(Hs):
                HASH_TABLES[i][H].add(key)
    print(f"Number of clusters: {len(HASH_TABLES)}")
    for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
        # cluster: Set[int]
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                edges.append((x, idx))
                uf.union(x, idx)
    print(f"Number of edges: {len(set(edges))}")

    ds = ds.map(
        function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
        with_indices=False,
        num_proc=num_proc,
        new_fingerprint=str(random.getrandbits(128)),
        desc="Finding clusters...",
    )

    final_data = ds.filter(
        function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
        with_indices=False,
        num_proc=num_proc,
        desc="Filtering clusters...",
    )

    return final_data.remove_columns([CLUSTER_COLUMN, INDEX_COLUMN])


if __name__ == "main":
    ds=load_dataset("fhai50032/HINGLISH-LIMA",split="train")
    column="Hinglish"
    print(deduplicate_dataset(ds,column))
