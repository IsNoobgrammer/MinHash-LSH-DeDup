# MinHash-LSH
Batched Minhash-LSH deduplication for large datasets
This Repo is Highly Adopted from datasketch and text-dedup
by default uses SHA-64 for hash signature

to get started


```
git clone https://github.com/IsNoobgrammer/MinHash-LSH-DeDup.git
cd MinHash-LSH-DeDup
pip install scipy datasets torch hf_transfer -qU
export HF_HUB_ENABLE_HF_TRANSFER=1
```

To use 
```
from datasets import load_dataset
from LSH import deduplicate_dataset

ds=load_dataset("fhai50032/HINGLISH-LIMA",split="train")
column="Hinglish"

dedup_ds=deduplicate_dataset(ds,column)
```

args
```
    ds: Dataset,
    column,
    threshold=0.8, 
    num_perm=256,
    batch_size=10_000,
    num_proc= 1 if os.name == "nt" else os.cpu_count() ,
    ngram_size=5,
    min_length=5,
    bands_rows=(None,None)
```
