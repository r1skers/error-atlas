"""Output bookkeeping and storage casts for the bounded scalar V diagnostic.

Uses the user's numerator/division/reference cores. No relative error against y is
formed: y may be zero. Casts start from the actual FP32 result, never from a float
approximation of the real reference. Fraction canonicalizes both zero signs.
"""

from fractions import Fraction
import struct

from .attribution_runner import absolute_interval
from .fp32_signed import is_stored_fp32
from .output_reference import quotient_interval
from .pilot_runner import fp32_word

DTYPES = ("fp32", "fp16", "bf16")


def storage_cast(value, dtype):
    """RN-even FP32 -> storage dtype, gradual underflow, canonical +0.

    Restrict to [-2,2], which covers this scalar |V|<=1 diagnostic with ample
    headroom. This is not a general overflow/NaN/signed-zero implementation.
    Binary64 represents every FP32 input exactly; struct's binary16 conversion is
    one rounding. BF16 retains the high 16 FP32 bits with round/sticky/tie parity.
    """
    if not is_stored_fp32(value) or abs(value) > 2:
        raise ValueError("Storage cast requires a stored FP32 value in [-2,2]")
    if dtype == "fp32":
        return value, fp32_word(value)
    if dtype == "fp16":
        encoded = struct.pack(">e", float(value))
        result = Fraction(struct.unpack(">e", encoded)[0])
        return result, encoded.hex() if result else "0000"
    if dtype == "bf16":
        bits = int(fp32_word(value), 16)
        upper = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
        result = Fraction(struct.unpack(">f", struct.pack(">I", upper << 16))[0])
        return result, f"{upper:04x}" if result else "0000"
    raise ValueError(f"Unknown output dtype: {dtype}")


def difference(left, right):
    return left[0] - right[1], left[1] - right[0]


def record(computed, reference, scale):
    """Signed errors, final division residual, and the two one-sided replacements.

    The counterfactuals are diagnostics, not outputs of a deployable kernel. Their
    errors are not independent and must not be added as absolute error budgets.
    """
    if scale < 0:
        raise ValueError("Output normalization scale must be nonnegative")
    formats = {}
    for dtype in DTYPES:
        value, bits = storage_cast(computed.value, dtype)
        signed = value - reference.output[1], value - reference.output[0]
        error = absolute_interval(signed)
        formats[dtype] = {"value": value, "bits": bits, "signed_error": signed,
                          "abs_error": error,
                          "normalized_abs_error": tuple(v / scale for v in error) if scale else None}
    only_denominator = quotient_interval(reference.numerator, (computed.denominator, computed.denominator))
    only_numerator = quotient_interval((computed.numerator, computed.numerator), reference.denominator)
    return {"numerator": computed.numerator, "denominator": computed.denominator,
            "before_division_rounding": computed.before_division_rounding,
            "division_residual": computed.division_residual,
            "delta_O": (computed.numerator - reference.numerator[1], computed.numerator - reference.numerator[0]),
            "delta_ell": (computed.denominator - reference.denominator[1], computed.denominator - reference.denominator[0]),
            "only_denominator": only_denominator, "only_numerator": only_numerator,
            "denominator_only_abs_error": absolute_interval(difference(only_denominator, reference.output)),
            "formats": formats}


def paired(chain, balanced):
    """Same-reference error improvement and actual output changes for each dtype."""
    result = {}
    for dtype in DTYPES:
        c, b = chain["formats"][dtype], balanced["formats"][dtype]
        equal = c["bits"] == b["bits"]
        result[dtype] = {"outputs_equal": equal, "output_delta": c["value"] - b["value"],
                         "D": (Fraction(0), Fraction(0)) if equal else difference(c["abs_error"], b["abs_error"])}
    return result
