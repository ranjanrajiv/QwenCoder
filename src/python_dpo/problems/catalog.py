"""The manually curated problem catalog.

``build_catalog`` is pure and deterministic: same source, same dataset, every time.
No timestamps, no generated ids, no randomness (spec 02 section 34).

Each prompt states the behavior that would otherwise be ambiguous — tie-breaking,
output ordering, and invalid-input handling — because Stage 3 candidates are graded
against exactly these rules.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from . import references
from .models import Problem, TestCase


def _source(function: Callable[..., Any]) -> str:
    """Return a reference solution's source text, kept in step with the real code."""
    return inspect.getsource(function)


def _case(
    identifier: str,
    inputs: dict[str, Any],
    expected: Any = None,
    *,
    raises: str | None = None,
) -> TestCase:
    return TestCase(id=identifier, input=inputs, expected=expected, expected_exception=raises)


def build_catalog() -> list[Problem]:
    """Return the curated problems in stable id order."""
    return [
        _p001_sum_even(),
        _p002_most_frequent_value_key(),
        _p003_first_non_repeating(),
        _p004_common_elements(),
        _p005_k_largest(),
        _p006_factorial(),
        _p007_safe_get(),
        _p008_parse_int(),
        _p009_chunk_sequence(),
        _p010_gather_in_order(),
    ]


def _p001_sum_even() -> Problem:
    return Problem(
        id="p001",
        category="lists",
        difficulty="easy",
        entry_point="sum_even",
        signature="def sum_even(numbers):",
        prompt=(
            "Write a function that returns the sum of all even integers in a list.\n\n"
            "Rules:\n"
            "- An empty list returns 0.\n"
            "- Negative even numbers are included in the sum.\n"
            "- Duplicate values are each counted.\n"
            "- Zero is even and contributes 0."
        ),
        description="Sum the even integers in a list.",
        tags=("iteration", "arithmetic"),
        reference_solution=_source(references.sum_even),
        tests=(
            _case("t001", {"numbers": [1, 2, 3, 4]}, 6),
            _case("t002", {"numbers": []}, 0),
            _case("t003", {"numbers": [1, 3, 5, 7]}, 0),
            _case("t004", {"numbers": [-2, -4, 3]}, -6),
            _case("t005", {"numbers": [2, 2, 2]}, 6),
            _case("t006", {"numbers": [0, 1, 3]}, 0),
            _case("t007", {"numbers": [-1, 0, 1, 2]}, 2),
        ),
    )


def _p002_most_frequent_value_key() -> Problem:
    return Problem(
        id="p002",
        category="dictionaries",
        difficulty="medium",
        entry_point="most_frequent_value_key",
        signature="def most_frequent_value_key(mapping):",
        prompt=(
            "Write a function that returns the key whose value occurs most frequently "
            "in a dictionary.\n\n"
            "Rules:\n"
            "- Count how often each value appears across the dictionary.\n"
            "- Return a key holding one of the most frequent values.\n"
            "- If several keys qualify, return the one that comes first in the "
            "dictionary's insertion order.\n"
            "- An empty dictionary returns None."
        ),
        description="Find the key whose value is most common, breaking ties by insertion order.",
        tags=("counting", "ordering"),
        reference_solution=_source(references.most_frequent_value_key),
        tests=(
            _case("t001", {"mapping": {"a": 1, "b": 2, "c": 2}}, "b"),
            _case("t002", {"mapping": {"only": 9}}, "only"),
            _case("t003", {"mapping": {"a": 1, "b": 2}}, "a"),
            _case("t004", {"mapping": {}}, None),
            _case("t005", {"mapping": {"x": 5, "y": 5, "z": 5}}, "x"),
            _case("t006", {"mapping": {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2}}, "c"),
            _case("t007", {"mapping": {"a": 1, "b": 1, "c": 2, "d": 2}}, "a"),
        ),
    )


def _p003_first_non_repeating() -> Problem:
    return Problem(
        id="p003",
        category="strings",
        difficulty="medium",
        entry_point="first_non_repeating",
        signature="def first_non_repeating(text):",
        prompt=(
            "Write a function that returns the first character in a string that occurs "
            "exactly once.\n\n"
            "Rules:\n"
            "- Return None when every character occurs more than once.\n"
            "- Return None for an empty string.\n"
            "- Comparison is case-sensitive: 'a' and 'A' are different characters.\n"
            "- Preserve the original left-to-right order when deciding which "
            "non-repeating character comes first."
        ),
        description="Find the first character that appears exactly once.",
        tags=("counting", "ordering"),
        reference_solution=_source(references.first_non_repeating),
        tests=(
            _case("t001", {"text": "abca"}, "b"),
            _case("t002", {"text": ""}, None),
            _case("t003", {"text": "a"}, "a"),
            _case("t004", {"text": "aabb"}, None),
            _case("t005", {"text": "swiss"}, "w"),
            _case("t006", {"text": "aA"}, "a"),
            _case("t007", {"text": "aabbc"}, "c"),
        ),
    )


def _p004_common_elements() -> Problem:
    return Problem(
        id="p004",
        category="sets",
        difficulty="medium",
        entry_point="common_elements",
        signature="def common_elements(first, second):",
        prompt=(
            "Write a function that returns the elements appearing in both of two "
            "collections.\n\n"
            "Rules:\n"
            "- Return a list, not a set.\n"
            "- Each common element appears exactly once, even if it repeats in the input.\n"
            "- Order the result by first appearance in the first collection.\n"
            "- Return an empty list when either collection is empty or nothing is shared."
        ),
        description="Intersect two collections, deduplicated and ordered by the first input.",
        tags=("membership", "ordering"),
        reference_solution=_source(references.common_elements),
        tests=(
            _case("t001", {"first": [1, 2, 3], "second": [2, 3, 4]}, [2, 3]),
            _case("t002", {"first": [3, 1, 2], "second": [1, 2, 3]}, [3, 1, 2]),
            _case("t003", {"first": [1, 1, 2], "second": [1, 2]}, [1, 2]),
            _case("t004", {"first": [], "second": [1]}, []),
            _case("t005", {"first": [1], "second": []}, []),
            _case("t006", {"first": [1, 2], "second": [3, 4]}, []),
            _case("t007", {"first": [1, 2], "second": [2, 2, 2]}, [2]),
            _case("t008", {"first": ["b", "a"], "second": ["a", "b"]}, ["b", "a"]),
        ),
    )


def _p005_k_largest() -> Problem:
    return Problem(
        id="p005",
        category="sorting",
        difficulty="medium",
        entry_point="k_largest",
        signature="def k_largest(values, k):",
        prompt=(
            "Write a function that returns the k largest values from a list.\n\n"
            "Rules:\n"
            "- Duplicates are preserved: k_largest([5, 1, 5, 3], 3) returns [5, 5, 3].\n"
            "- The result is ordered from largest to smallest.\n"
            "- k == 0 returns an empty list.\n"
            "- k greater than the list length returns every value, still sorted "
            "descending.\n"
            "- A negative k raises ValueError."
        ),
        description="Return the k largest values, descending, keeping duplicates.",
        tags=("sorting", "slicing"),
        reference_solution=_source(references.k_largest),
        tests=(
            _case("t001", {"values": [5, 1, 5, 3], "k": 3}, [5, 5, 3]),
            _case("t002", {"values": [5, 1, 5, 3], "k": 0}, []),
            _case("t003", {"values": [1, 2], "k": 5}, [2, 1]),
            _case("t004", {"values": [], "k": 3}, []),
            _case("t005", {"values": [1, 2, 3], "k": -1}, raises="ValueError"),
            _case("t006", {"values": [4, 4, 4], "k": 2}, [4, 4]),
            _case("t007", {"values": [3, 9, 2], "k": 1}, [9]),
            _case("t008", {"values": [-5, -1, -3], "k": 2}, [-1, -3]),
        ),
    )


def _p006_factorial() -> Problem:
    return Problem(
        id="p006",
        category="recursion",
        difficulty="easy",
        entry_point="factorial",
        signature="def factorial(n):",
        prompt=(
            "Write a recursive function that returns the factorial of a non-negative "
            "integer.\n\n"
            "Rules:\n"
            "- factorial(0) and factorial(1) both return 1.\n"
            "- factorial(n) returns n * factorial(n - 1) for larger values.\n"
            "- A negative input raises ValueError.\n"
            "- The implementation must be recursive rather than an iterative loop."
        ),
        description="Compute n! recursively, rejecting negative input.",
        tags=("recursion", "validation"),
        reference_solution=_source(references.factorial),
        tests=(
            _case("t001", {"n": 0}, 1),
            _case("t002", {"n": 1}, 1),
            _case("t003", {"n": 2}, 2),
            _case("t004", {"n": 5}, 120),
            _case("t005", {"n": 10}, 3628800),
            _case("t006", {"n": -1}, raises="ValueError"),
            _case("t007", {"n": -5}, raises="ValueError"),
        ),
    )


def _p007_safe_get() -> Problem:
    return Problem(
        id="p007",
        category="edge_cases",
        difficulty="easy",
        entry_point="safe_get",
        signature="def safe_get(items, index, default=None):",
        prompt=(
            "Write a function that retrieves an element from a list by index and "
            "returns a fallback value when the index cannot be used.\n\n"
            "Rules:\n"
            "- A valid index returns the element at that position.\n"
            "- Negative indices keep their usual Python meaning: -1 is the last element.\n"
            "- An index beyond either end of the list returns default.\n"
            "- An index that is not an integer returns default rather than raising.\n"
            "- An empty list always returns default.\n"
            "- default is None unless the caller supplies another value."
        ),
        description="Index into a list without raising, falling back to a default.",
        tags=("indexing", "defaults"),
        reference_solution=_source(references.safe_get),
        tests=(
            _case("t001", {"items": [], "index": 0}, None),
            _case("t002", {"items": [10, 20, 30], "index": 0}, 10),
            _case("t003", {"items": [10, 20, 30], "index": 2}, 30),
            _case("t004", {"items": [10, 20, 30], "index": -1}, 30),
            _case("t005", {"items": [10, 20, 30], "index": 5}, None),
            _case("t006", {"items": [10, 20, 30], "index": -4}, None),
            _case("t007", {"items": [1, 2], "index": 9, "default": "missing"}, "missing"),
            _case("t008", {"items": [1, 2, 3], "index": "1"}, None),
        ),
    )


def _p008_parse_int() -> Problem:
    return Problem(
        id="p008",
        category="exceptions",
        difficulty="easy",
        entry_point="parse_int",
        signature="def parse_int(text, default=0):",
        prompt=(
            "Write a function that converts a string to an integer and returns a "
            "default value when the conversion fails.\n\n"
            "Rules:\n"
            "- Valid integer text converts normally, including negative values.\n"
            "- A leading '+' or '-' sign is accepted.\n"
            "- Surrounding whitespace is ignored: '  42  ' converts to 42.\n"
            "- Text that is not an integer, including '3.5' and the empty string, "
            "returns default.\n"
            "- None returns default rather than raising.\n"
            "- The function never raises; default is 0 unless the caller supplies one."
        ),
        description="Parse an integer defensively, falling back to a default.",
        tags=("exceptions", "parsing"),
        reference_solution=_source(references.parse_int),
        tests=(
            _case("t001", {"text": "42"}, 42),
            _case("t002", {"text": "-7"}, -7),
            _case("t003", {"text": "+5"}, 5),
            _case("t004", {"text": "  42  "}, 42),
            _case("t005", {"text": "abc"}, 0),
            _case("t006", {"text": ""}, 0),
            _case("t007", {"text": None}, 0),
            _case("t008", {"text": "3.5"}, 0),
            _case("t009", {"text": "abc", "default": -1}, -1),
        ),
    )


def _p009_chunk_sequence() -> Problem:
    return Problem(
        id="p009",
        category="generators",
        difficulty="easy",
        entry_point="chunk_sequence",
        signature="def chunk_sequence(sequence, size):",
        prompt=(
            "Write a generator that yields consecutive chunks of a sequence.\n\n"
            "Rules:\n"
            "- Each chunk is a list of up to size items, in the original order.\n"
            "- The final chunk may be shorter: "
            "list(chunk_sequence([1, 2, 3, 4, 5], 2)) returns [[1, 2], [3, 4], [5]].\n"
            "- An empty sequence yields nothing.\n"
            "- The function must be a generator that yields chunks one at a time, not "
            "a function that builds and returns the whole list.\n"
            "- A size of zero or less raises ValueError. Because the function is a "
            "generator, that error surfaces when iteration begins."
        ),
        description="Yield fixed-size chunks of a sequence, last chunk possibly shorter.",
        tags=("generators", "batching"),
        reference_solution=_source(references.chunk_sequence),
        tests=(
            _case("t001", {"sequence": [1, 2, 3, 4, 5], "size": 2}, [[1, 2], [3, 4], [5]]),
            _case("t002", {"sequence": [1, 2, 3], "size": 1}, [[1], [2], [3]]),
            _case("t003", {"sequence": [1, 2], "size": 5}, [[1, 2]]),
            _case("t004", {"sequence": [], "size": 2}, []),
            _case("t005", {"sequence": [1, 2, 3, 4], "size": 2}, [[1, 2], [3, 4]]),
            _case("t006", {"sequence": [1, 2, 3], "size": 0}, raises="ValueError"),
            _case("t007", {"sequence": [1, 2, 3], "size": -1}, raises="ValueError"),
        ),
    )


def _p010_gather_in_order() -> Problem:
    return Problem(
        id="p010",
        category="async",
        difficulty="hard",
        entry_point="gather_in_order",
        signature="async def gather_in_order(operations):",
        prompt=(
            "Write an asynchronous function that runs several asynchronous operations "
            "concurrently and collects their results.\n\n"
            "operations is a list of [value, delay] pairs. Each operation waits for its "
            "delay in seconds and then produces its value.\n\n"
            "Rules:\n"
            "- Run the operations concurrently, not one after another; the total time "
            "should be close to the longest delay rather than the sum of all delays.\n"
            "- Return a list of the values in the order they appear in operations, not "
            "the order in which the operations finished.\n"
            "- An empty list of operations returns an empty list.\n"
            "- Use only local coroutines and asyncio; no network or external services."
        ),
        description="Run simulated async operations concurrently, returning results in input order.",
        tags=("async", "concurrency", "ordering"),
        reference_solution=_source(references.gather_in_order),
        tests=(
            _case("t001", {"operations": [[1, 0.03], [2, 0.01], [3, 0.02]]}, [1, 2, 3]),
            _case("t002", {"operations": [[7, 0.01]]}, [7]),
            _case("t003", {"operations": []}, []),
            _case("t004", {"operations": [[1, 0.01], [2, 0.01]]}, [1, 2]),
            _case("t005", {"operations": [["a", 0.03], ["b", 0.02], ["c", 0.01]]}, ["a", "b", "c"]),
            _case("t006", {"operations": [[1, 0], [2, 0]]}, [1, 2]),
        ),
    )
