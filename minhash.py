"""
Adopted from text-dedup && datasketch
"""

from utils import ngrams

import re
from typing import Any
from typing import Callable
import datasets
import numpy as np


SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()

SIGNATURE_COLUMN = "__signatures__"
INDEX_COLUMN = "__index__"


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: list[tuple[int, int]],
    permutations: np.ndarray,
    hash_func: Callable,
    dtype: type,
    max_hash: np.uint,
    modulo_prime: np.uint,
) -> dict[str, Any]:
    """
    Calculate hash values for the content.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    min_length : int
        The minimum length of the document in terms of tokens.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    hash_func : Callable
        The hash function to use.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.

"""
    # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
    # the formula is a * x(base hash of each shingle) + b
    a, b = permutations
    # split content on whitespace (NON_ALPHA regex), tokenize with ngrams(), and join these n-grams into a single space separated string.
    # we then convert to lower case and then bytestrings which is then hashed. Only unique hashed n-grams are left.
    tokens: set[bytes] = {
        bytes(" ".join(t).lower(), "utf-8") for t in ngrams(NON_ALPHA.split(str(content).lower()), ngram_size, min_length)
    }

    hashvalues: np.ndarray = np.array([hash_func(token) for token in tokens], dtype=dtype).reshape(len(tokens), 1)
    # Permute the hash values to produce new universal hashes
    # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
    # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
    hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
    # this part is where the name "min" of minhash comes from
    # this stacks all the hashes and then takes the minimum from each column
    masks: np.ndarray = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
    # Originally, byteswap was done for speed. Testing show it has a negligible impact
    # keeping  for backward compatibility, even though theoretically and empirically
    # it doesnt matter if it is there or not. github.com/ekzhu/datasketch/issues/114
    Hs: list[bytes] = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {SIGNATURE_COLUMN: Hs, INDEX_COLUMN: idx}








