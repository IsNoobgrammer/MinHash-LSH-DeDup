import hashlib
import struct

def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    >>> sha1_hash(b"hello world", 128)
    310522945683037930239412421226792791594
    """
    if d == 32:
        return struct.unpack("<I", hashlib.sha1(data, usedforsecurity=False).digest()[:4])[0]
    if d == 64:
        return struct.unpack("<Q", hashlib.sha1(data, usedforsecurity=False).digest()[:8])[0]
    # struct is faster but does not support arbitrary bit lengths
    return int.from_bytes(hashlib.sha1(data, usedforsecurity=False).digest()[: d // 8], byteorder="little")

if __name__ == "__main__":
    print(sha1_hash(b"Hello World!",64))


