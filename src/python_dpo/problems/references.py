"""Trusted reference solutions for the curated problems.

Every function here is manually authored, deterministic, and free of I/O, network
access, environment lookups, and third-party imports. These are the correctness
oracle for the dataset — not the "preferred" DPO answer (see spec 02 section 12).

This module is the *only* source of code the validator executes. Model-generated
candidates must never be routed through the reference executor; they belong in the
Stage 3 sandbox.

Each function is stored in the dataset via ``inspect.getsource``, so every solution
must stand alone: any import it needs lives inside the function body, exactly as a
standalone candidate implementation would.
"""

from __future__ import annotations


def sum_even(numbers):
    """Return the sum of the even integers in ``numbers`` (0 for an empty list)."""
    return sum(number for number in numbers if number % 2 == 0)


def most_frequent_value_key(mapping):
    """Return the key whose value occurs most often, or None for an empty mapping.

    Ties are broken by insertion order: the first key holding a most-frequent value.
    """
    if not mapping:
        return None

    counts = {}
    for value in mapping.values():
        counts[value] = counts.get(value, 0) + 1

    highest = max(counts.values())
    for key, value in mapping.items():
        if counts[value] == highest:
            return key
    return None


def first_non_repeating(text):
    """Return the first character occurring exactly once, or None if there is none."""
    counts = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1

    for character in text:
        if counts[character] == 1:
            return character
    return None


def common_elements(first, second):
    """Return elements present in both collections, deduplicated.

    Results follow the order of first appearance in ``first``.
    """
    lookup = set(second)
    seen = set()
    result = []
    for item in first:
        if item in lookup and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def k_largest(values, k):
    """Return the ``k`` largest values in descending order, keeping duplicates."""
    if k < 0:
        raise ValueError("k must not be negative")
    return sorted(values, reverse=True)[:k]


def factorial(n):
    """Return n! computed recursively; a negative ``n`` raises ValueError."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n in (0, 1):
        return 1
    return n * factorial(n - 1)


def safe_get(items, index, default=None):
    """Return ``items[index]``, or ``default`` when the index is not usable.

    Negative indices keep their normal Python meaning; out-of-range and non-integer
    indices fall back to ``default``.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        return default
    try:
        return items[index]
    except IndexError:
        return default


def parse_int(text, default=0):
    """Return ``text`` converted to an int, or ``default`` when conversion fails."""
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def chunk_sequence(sequence, size):
    """Yield successive lists of up to ``size`` items from ``sequence``.

    This is a generator function, so a non-positive ``size`` raises ValueError when
    iteration begins rather than at call time.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")

    chunk = []
    for item in sequence:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


async def gather_in_order(operations):
    """Run every operation concurrently and return the values in input order.

    ``operations`` is a sequence of ``[value, delay]`` pairs. Each operation waits for
    its delay and then produces its value; the returned list follows the input order,
    not the order in which the operations finished.
    """
    import asyncio

    async def run_operation(value, delay):
        await asyncio.sleep(delay)
        return value

    results = await asyncio.gather(
        *(run_operation(value, delay) for value, delay in operations)
    )
    return list(results)
