"""Human-readable word-based hashes using coolname.

Deterministic: same input always produces the same word pair.
"""

from __future__ import annotations

import hashlib
import random
import threading

import coolname

# coolname.replace_random swaps library-global state; serialize callers so
# concurrent wordhash() calls (threads in cp.map / notify) can't interleave
# the swap-generate-restore sequence.
_coolname_lock = threading.Lock()


def wordhash(data: str | bytes, words: int = 2) -> str:
    """Map data to a memorable word slug like 'gentle-fox'.

    Deterministic -- same input always produces the same output.
    Uses coolname's curated word lists (4.6M+ combinations for 2 words).
    """
    if isinstance(data, str):
        data = data.encode()
    seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
    rng = random.Random(seed)
    with _coolname_lock:
        coolname.replace_random(rng)
        try:
            return coolname.generate_slug(words)
        finally:
            coolname.replace_random(random)
