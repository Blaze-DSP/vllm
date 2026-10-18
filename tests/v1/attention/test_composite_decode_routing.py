# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing rule of the TRITON_DECODE_FLASH_ATTN composite backend."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.composite import (
    MMPrefixAttentionRouting,
    MMPrefixOrDecodeAttentionRouting,
)

TRITON, FLASH_ATTN = 0, 1


def _config(sliding_window=None, clamped=False, model_config=None):
    layer = SimpleNamespace(
        sliding_window=sliding_window, mm_prefix_clamp_sliding_window=clamped
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"attn": layer}),
        model_config=model_config,
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


TEXT_ONLY = SimpleNamespace(is_multimodal_model=False)
LANGUAGE_MODEL_ONLY = SimpleNamespace(
    is_multimodal_model=True,
    multimodal_config=SimpleNamespace(language_model_only=True),
)
WITH_IMAGES = SimpleNamespace(
    is_multimodal_model=True,
    multimodal_config=SimpleNamespace(language_model_only=False),
)


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        # No image can reach a batch: routing is a function of the batch shape,
        # so spec-decode verify batches (1 + K queries) can keep FULL graphs.
        (TEXT_ONLY, AttentionCGSupport.UNIFORM_BATCH),
        (LANGUAGE_MODEL_ONLY, AttentionCGSupport.UNIFORM_BATCH),
        # Image ranges make routing data-dependent: keep the base level.
        (WITH_IMAGES, AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE),
    ],
)
def test_decode_routing_graph_support(model_config, expected):
    config = _config(sliding_window=128, clamped=True, model_config=model_config)
    support = MMPrefixOrDecodeAttentionRouting.get_cudagraph_support(config, None)
    assert support == expected
    assert MMPrefixAttentionRouting.get_cudagraph_support(config, None) == (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_unclamped_window_still_vetoes_graphs():
    # A layer that does not clamp its window to the image span (as the Gemma 4
    # MTP drafter's sliding layers did) must keep vetoing graphs.
    config = _config(sliding_window=128, clamped=False, model_config=TEXT_ONLY)
    support = MMPrefixOrDecodeAttentionRouting.get_cudagraph_support(config, None)
    assert support == AttentionCGSupport.NEVER
