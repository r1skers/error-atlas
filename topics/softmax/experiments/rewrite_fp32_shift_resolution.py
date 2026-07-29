"""Closed-book rewrite of the FP32 Softmax boundary probe.

Complete this file without importing or copying the first implementation.
The objective is to recover the quantize-before-stabilize mechanism from the
research invariants rather than from surface syntax.
"""

import numpy as np


def rewritten_fp32_softmax_probe(common_offset: float) -> tuple[float, float]:
    """Return the stored unit gap and first stable-Softmax probability.

    Preserve these invariants:
    - the ideal logits are ``(M + 1, M)``;
    - input quantization happens before numerical stabilization;
    - the normalization cannot overflow for finite probe inputs;
    - all research arithmetic stays in FP32 until the return boundary.
    """
    logits = np.array([common_offset + 1.0, common_offset], dtype=np.float32)
    stored_difference = logits[0] - logits[1]
    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(exponentials)
    return float(stored_difference), float(probabilities[0])
