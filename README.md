# MinHash-LSH
Batched Minhash-LSH deduplication for large datasets
This Repo is Highly Adopted from datasketch and text-dedup

to get started


```
git clone https://github.com/IsNoobgrammer/MinHash-LSH.git
cd MinHash-LSH
pip install scipy datasets torch -qU
```

To use 
```
from datasets import load_dataaset
from LSH import deduplicate_dataset

ds=load_dataset("fhai50032/HINGLISH-LIMA",split="train")
column="Hinglish"

dedup_ds=deduplicate_dataset(ds,column)
```
