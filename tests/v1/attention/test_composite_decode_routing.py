# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing rule of the TRITON_DECODE_FLASH_ATTN composite backend."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.composite import (
    MMPrefixAttentionRouting,
    MMPrefixOrDecodeAttentionRouting,
)

TRITON, FLASH_ATTN = 0, 1


def _config():
    layer = SimpleNamespace(sliding_window=None)
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"attn": layer})
    )


def _batch(num_reqs: int, max_query_len: int):
    return SimpleNamespace(
        num_reqs=num_reqs, max_query_len=max_query_len, mm_req_doc_ranges=None
    )


@pytest.mark.parametrize(
    ("num_reqs", "max_query_len", "expected"),
    [
        (1, 1, TRITON),
        (16, 1, TRITON),
        (17, 1, FLASH_ATTN),
        (256, 1, FLASH_ATTN),
        (1, 4, FLASH_ATTN),  # spec-decode verify batch
        (1, 512, FLASH_ATTN),  # prefill
    ],
)
def test_small_decode_batches_route_to_triton(num_reqs, max_query_len, expected):
    routing = MMPrefixOrDecodeAttentionRouting(["attn"], _config())
    assert routing.max_decode_reqs == 16
    metadata = _batch(num_reqs, max_query_len)
    assert routing.select(metadata) == expected
    # Capture must record the child that replay of this shape will select.
    assert routing.capture_select(metadata) == expected


def test_plain_mm_prefix_routing_captures_its_fixed_variant():
    routing = MMPrefixAttentionRouting(["attn"], _config())
    assert routing.select(_batch(1, 1)) == FLASH_ATTN
    assert routing.capture_select(_batch(1, 1)) == routing.capture_variant
