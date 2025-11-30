
from dataclasses import dataclass
from itertools import chain
from random import Random
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

Operation = Callable[[int, int], int]


@dataclass(frozen=True)
class ArithmeticDatasets:
    train: TensorDataset
    test: TensorDataset
    input_size: int
    target_size: int
    base: int
    operand_digits: int
    result_digits: int


def generate_arithmetic_datasets(
    train_min: int,
    train_max: int,
    test_min: int,
    test_max: int,
    operations: Sequence[Operation],
    base: int,
    *,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> ArithmeticDatasets:
    """
    Build train/test TensorDatasets encoding binary arithmetic problems.

    Features = sign(operand1) + base-N digits(operand1) +
                sign(operand2) + base-N digits(operand2) +
                single scalar op-code in [0, 1].

    Targets = sign(result) + base-N digits(result).
    The digit width is fixed by the largest operand/result magnitude across
both splits.
    """
    if base < 2:
        raise ValueError("base must be >= 2")
    if not operations:
        raise ValueError("operations must not be empty")
    if train_min > train_max or test_min > test_max:
        raise ValueError("min value must be <= max value for each split")

    in_max = max(abs(train_min), abs(train_max), abs(test_min), abs(test_max))

    ops = list(operations)
    operand_digits = _required_digits(
        in_max,
        base,
    )

    base_rng = Random(seed) if seed is not None else None
    train_rng = (
        Random(base_rng.getrandbits(64)) if base_rng is not None else None
    )
    test_rng = (
        Random(base_rng.getrandbits(64)) if base_rng is not None else None
    )

    train_examples = _collect_examples(
        train_min, train_max, ops, train_samples, train_rng
    )
    test_examples = _collect_examples(
        test_min, test_max, ops, test_samples, test_rng
    )

    out_max = max(
        0, *(
            abs(example[3]) for example in chain(train_examples, test_examples)
        ),
    )

    result_digits = _required_digits(
        out_max,
        base,
    )

    opcount = len(ops)

    train_inputs, train_targets = _encode_examples(
        train_examples, operand_digits, result_digits, base, opcount
    )
    test_inputs, test_targets = _encode_examples(
        test_examples, operand_digits, result_digits, base, opcount
    )

    print(f"\n\n{in_max=}, {out_max=}, {opcount=}")
    print(f"{base=}, {operand_digits=}, {result_digits=}")

    thing = ArithmeticDatasets(
        train=TensorDataset(train_inputs, train_targets),
        test=TensorDataset(test_inputs, test_targets),
        input_size=train_inputs.shape[1],
        target_size=train_targets.shape[1],
        base=base,
        operand_digits=operand_digits,
        result_digits=result_digits,
    )
    print(f"input_size={thing.input_size}, target_size={thing.target_size}")
    print(f"operand_digits={thing.operand_digits}, result_digits={thing.result_digits}")
    return thing

def _collect_examples(
    min_value: int,
    max_value: int,
    operations: Sequence[Operation],
    sample_count: Optional[int] = None,
    rng: Optional[Random] = None,
) -> List[Tuple[int, int, int, int]]:
    op_count = len(operations)
    if op_count == 0:
        return []
    if sample_count is None:
        span = max_value - min_value + 1
        sample_count = span * span * op_count
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    random_source = rng or Random()
    examples: List[Tuple[int, int, int, int]] = []
    for _ in range(sample_count):
        left = random_source.randint(min_value, max_value)
        right = random_source.randint(min_value, max_value)
        op_index = random_source.randrange(op_count)
        result = operations[op_index](left, right)
        examples.append((left, right, op_index, int(result)))
    return examples


def _required_digits(max_abs_value: int, base: int) -> int:
    digits = 1
    threshold = base
    while max_abs_value >= threshold:
        digits += 1
        threshold *= base

    return digits


def _encode_examples(
    examples: Sequence[Tuple[int, int, int, int]],
    operand_digits: int,
    result_digits: int,
    base: int,
    op_count: int,
) -> Tuple[Tensor, Tensor]:
    inputs: List[List[float]] = []
    targets: List[List[float]] = []
    op_denominator = max(1, op_count - 1)

    for left, right, op_index, result in examples:
        feature = (
            _encode_number(left, operand_digits, base)
            + _encode_number(right, operand_digits, base)
            + [op_index / op_denominator]
        )
        target = _encode_number(result, result_digits, base)
        inputs.append(feature)
        targets.append(target)

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def _encode_number(value: int, digits: int, base: int) -> List[float]:
    sign = 1.0 if value < 0 else 0.0
    magnitude = abs(value)
    encoded_digits = [0.0] * digits
    for position in range(digits - 1, -1, -1):
        encoded_digits[position] = (magnitude % base) / (base - 1)
        magnitude //= base

    num = [sign] + encoded_digits
    # print(f"{value} -> {[round(n, 2) for n in num]}")
    return num
