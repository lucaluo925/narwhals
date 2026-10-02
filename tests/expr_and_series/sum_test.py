from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize("expr", [nw.col("a", "b", "z").sum(), nw.sum("a", "b", "z")])
def test_expr_sum_expr(constructor: Constructor, expr: nw.Expr) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    expected = {"a": [6], "b": [14], "z": [24.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 6), ("b", 14), ("z", 24.0)])
def test_expr_sum_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.sum()
    assert_equal_data({col: [result]}, {col: [expected]})


def test_expr_sum_integer_dtype(constructor: Constructor) -> None:
    # The sum of an integer column is an integer on every backend. DuckDB widens
    # `sum` to a 128-bit integer, which Arrow can only represent as a decimal.
    df = nw.from_native(constructor({"a": [1, 3, 2]}))
    result = df.select(nw.col("a").sum()).lazy().collect()
    assert result.collect_schema()["a"].is_integer()


def test_expr_sum_integer_dtype_group_by(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"g": ["x", "x", "y"], "a": [1, 3, 2]}))
    result = df.group_by("g").agg(nw.col("a").sum()).lazy().collect()
    assert result.collect_schema()["a"].is_integer()
