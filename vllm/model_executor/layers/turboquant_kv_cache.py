# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import lru_cache

import torch

# Use the vLLM root logger so messages always follow vLLM log config.
from vllm.logger import logger
import os


def _turboquant_debug_enabled() -> bool:
    # Keep this extremely cheap; called in hot paths.
    return os.environ.get("VLLM_TURBOQUANT_DEBUG", "0") not in ("", "0", "false", "False")


def parse_turboquant_kv_bits_key_value(
    additional_config: object | None,
) -> tuple[int, int]:
    """Read (key_bits, value_bits), each 4 or 8, from ``additional_config``.

    ``turboquant_kv_bits`` may be:
    - int: same width for K and V;
    - length-2 sequence: ``[key_bits, value_bits]``;
    - dict with ``key``/``k`` and ``value``/``v``.
    """
    if additional_config is None or not isinstance(additional_config, dict):
        return (4, 4)
    raw = additional_config.get("turboquant_kv_bits", 4)
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise ValueError(
                "additional_config['turboquant_kv_bits'] as list/tuple must "
                f"have length 2 [key_bits, value_bits], got {raw!r}"
            )
        k_bits = int(raw[0])
        v_bits = int(raw[1])
    elif isinstance(raw, dict):
        k_bits = int(raw.get("key", raw.get("k", 4)))
        v_bits = int(raw.get("value", raw.get("v", 4)))
    else:
        k_bits = v_bits = int(raw)
    for label, bits in ("key", k_bits), ("value", v_bits):
        if bits not in (4, 8):
            raise ValueError(
                "additional_config['turboquant_kv_bits'] "
                f"{label} must be 4 or 8, got {bits}"
            )
    return (k_bits, v_bits)


def parse_turboquant_kv_bits(additional_config: object | None) -> int:
    """Read key quant width (4 or 8). Prefer :func:`parse_turboquant_kv_bits_key_value` for both."""
    k_bits, _ = parse_turboquant_kv_bits_key_value(additional_config)
    return k_bits


def _debug_dump_packed(
    tag: str,
    packed: torch.Tensor,
    *,
    head_size: int,
    bits: int = 4,
    max_vecs: int = 1,
    max_bytes: int = 64,
) -> None:
    """Best-effort dump for TurboQuant packed bytes.

    This is intended for corruption/garbling debugging. It avoids calling
    TurboQuant dequantization to keep it usable even when a custom op produces
    `packed` tensors.
    """
    if not _turboquant_debug_enabled():
        return
    try:
        # Expect (..., P) where P=turboquant_packed_bytes_per_vector(head_size,bits)
        p = packed.reshape(-1, packed.shape[-1])
        n = min(int(p.shape[0]), int(max_vecs))
        per_vec = turboquant_packed_bytes_per_vector(head_size, bits=bits)
        logger.warning(
            "[TurboQuant debug] %s packed shape=%s dtype=%s device=%s per_vec=%d head_size=%d",
            tag,
            tuple(packed.shape),
            packed.dtype,
            packed.device,
            per_vec,
            head_size,
        )
        if p.numel() == 0 or n == 0:
            return

        # Dump first vectors' raw bytes (on CPU for readability).
        raw = p[:n].detach()
        if raw.is_cuda:
            raw = raw.cpu()
        raw = raw.contiguous().to(torch.uint8)
        nb = min(int(raw.shape[1]), int(max_bytes))
        for i in range(n):
            b = raw[i, :nb].tolist()
            # Norm bytes are the last 2 bytes; decode them as fp16 (little-endian).
            idx_len = head_size // 2 if bits == 4 else head_size
            norm_bytes = raw[i, idx_len : idx_len + 2].view(torch.uint8)
            norm_fp16 = norm_bytes.view(torch.float16).item()
            logger.warning(
                "[TurboQuant debug] %s vec[%d] first_bytes=%s norm_fp16=%s",
                tag,
                i,
                b,
                norm_fp16,
            )
    except Exception as e:
        logger.warning("[TurboQuant debug] %s dump failed: %r", tag, e)


def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8 bytes (nibble packing)."""
    if indices.dtype != torch.uint8:
        raise ValueError(f"indices must be uint8, got {indices.dtype}")
    if indices.shape[-1] % 2 != 0:
        raise ValueError(f"Last dim must be even, got {indices.shape[-1]}")
    high = indices[..., 0::2] << 4
    low = indices[..., 1::2] & 0x0F
    return (high | low).to(torch.uint8)


def unpack_uint4(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes into 4-bit indices (uint8 values 0..15)."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}")
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    out = torch.empty(
        packed.shape[:-1] + (orig_dim,), dtype=torch.uint8, device=packed.device
    )
    out[..., 0::2] = high
    out[..., 1::2] = low
    return out


def _beta_codebook_via_sampling(
    *,
    dim: int,
    bits: int,
    device: torch.device,
    seed: int,
    n_samples: int = 200_000,
    n_iters: int = 30,
) -> torch.Tensor:
    """
    Approximate the optimal 1D scalar quantizer (codebook) for coordinates of a
    randomly rotated unit vector in R^dim.

    Paper distribution: each coordinate has a symmetric Beta distribution on
    [-1, 1], obtained by mapping Beta(alpha, alpha) on [0,1] with
    alpha = (dim-1)/2.

    We avoid scipy by sampling and running a 1D Lloyd-Max / k-means procedure.
    This is computed once per (dim, bits) and cached.
    """
    if bits < 1 or bits > 4:
        raise NotImplementedError("vLLM TurboQuant currently supports bits=1..4.")
    k = 1 << bits
    alpha = (dim - 1) / 2.0
    # Sample on CPU (fast) then move to target device.
    # NOTE: torch.distributions.*.sample does not consistently accept a
    # `generator=` kwarg across torch versions, so use fork_rng+manual_seed for
    # determinism.
    beta = torch.distributions.Beta(alpha, alpha)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        u = beta.sample((n_samples,))  # [0,1]
    x = (2.0 * u - 1.0).to(dtype=torch.float32, device=device)  # [-1,1]
    x = x.clamp(-1.0, 1.0)

    # Initialize centroids with symmetric quantiles (approx).
    # Use evenly spaced probabilities to seed.
    probs = (torch.arange(1, k + 1, device=device, dtype=torch.float32) - 0.5) / k
    # torch.quantile expects float CPU for some builds; keep on device but fallback if needed.
    try:
        c = torch.quantile(x, probs).contiguous()
    except Exception:
        c = torch.quantile(x.cpu(), probs.cpu()).to(device=device).contiguous()

    # Lloyd iterations: assign to nearest centroid, then update mean.
    for _ in range(n_iters):
        # [N, K]
        d = (x[:, None] - c[None, :]).abs()
        a = d.argmin(dim=1)  # [N]
        new_c = torch.empty_like(c)
        for j in range(k):
            mask = a == j
            if torch.any(mask):
                new_c[j] = x[mask].mean()
            else:
                new_c[j] = c[j]
        # Enforce sorted + symmetry for stability.
        new_c, _ = torch.sort(new_c)
        c = new_c
    return c


class TurboQuantMSE:
    """
    Minimal TurboQuantMSE implementation vendored into vLLM.

    - Random orthogonal rotation via QR on a Gaussian matrix (det=+1 enforced).
    - Per-coordinate scalar quantization using a codebook derived from the
      theoretical coordinate distribution (approximated by sampling).
    - Stores a single norm per vector (float32) for rescaling.
    """

    def __init__(self, *, dim: int, bits: int = 4, device: torch.device, seed: int = 42):
        if bits == 4 and dim % 2 != 0:
            # For 4-bit nibble packing.
            raise ValueError(f"TurboQuant 4-bit requires even dim, got {dim}.")

        self.dim = dim
        self.bits = bits
        self.num_centroids = 1 << bits
        self.device = device
        self.seed = seed

        # Rotation matrix (dim, dim) computed on CPU deterministically.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        g = torch.randn((dim, dim), generator=gen, dtype=torch.float32, device="cpu")
        q, r = torch.linalg.qr(g)
        # Ensure det = +1 by normalizing column signs using diag(r).
        s = torch.sign(torch.diag(r))
        s[s == 0] = 1
        q = q * s.unsqueeze(0)
        self.rotation = q.to(device=device)
        self.rotation_t = q.T.to(device=device)

        # Codebook (K,)
        self.codebook = _beta_codebook_via_sampling(
            dim=dim, bits=bits, device=device, seed=seed + 1234
        )

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, dim] float tensor.
        Returns:
            indices: [N, dim] uint8 in 0..(2^bits-1)
            norms: [N, 1] float32
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        x_f32 = x.to(dtype=torch.float32)
        norms = torch.linalg.vector_norm(x_f32, dim=-1, keepdim=True)
        x_unit = x_f32 / (norms + 1e-10)
        # Rotate into coordinate system where per-dim scalar quantization is near-optimal.
        y = x_unit @ self.rotation_t
        # Assign nearest centroid per coordinate.
        # [N, dim, K]
        d = (y.unsqueeze(-1) - self.codebook.view(1, 1, -1)).abs()
        idx = d.argmin(dim=-1).to(torch.uint8)
        return idx, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {indices.shape[-1]}")
        y_hat = self.codebook[indices.to(torch.int64)]
        x_hat = y_hat @ self.rotation
        return x_hat * norms.to(dtype=torch.float32)


@lru_cache(maxsize=32)
def _get_quantizer(dim: int, bits: int, device: str) -> object:
    # Use a deterministic seed so cache contents are reproducible.
    return TurboQuantMSE(dim=dim, bits=bits, device=torch.device(device), seed=42)


def turboquant_packed_bytes_per_vector(head_size: int, bits: int = 4) -> int:
    """Bytes per head vector: packed index bytes + 2-byte fp16 norm slot."""
    if bits not in (4, 8):
        raise ValueError(f"TurboQuant KV supports bits 4 or 8 only, got {bits}.")
    if bits == 4:
        if head_size % 2 != 0:
            raise ValueError(
                "TurboQuant 4-bit uses nibble packing; head_size must be "
                f"even, got {head_size}."
            )
        return head_size // 2 + 2
    return head_size + 2


def turboquant_quantize_to_packed_bytes(
    x: torch.Tensor, *, bits: int = 4
) -> torch.Tensor:
    """
    Quantize vectors with TurboQuantMSE and pack into a uint8 byte tensor.

    Args:
        x: (..., head_size) float tensor.
    Returns:
        packed: (..., P) uint8 with P from ``turboquant_packed_bytes_per_vector``;
        last 2 bytes store fp16 norm.
    """
    head_size = x.shape[-1]
    logger.debug(
        "TurboQuant quantize: x=%s dtype=%s device=%s bits=%d head_size=%d",
        tuple(x.shape),
        x.dtype,
        x.device,
        bits,
        head_size,
    )
    packed_bytes = turboquant_packed_bytes_per_vector(head_size, bits=bits)
    device_str = str(x.device)
    quantizer = _get_quantizer(head_size, bits, device_str)

    x_flat = x.reshape(-1, head_size)
    indices, norms = quantizer.quantize(x_flat)  # indices: uint8 [N,D], norms: f32 [N,1]
    if bits == 4:
        indices_packed = pack_uint4(indices)
    elif bits == 8:
        indices_packed = indices.to(dtype=torch.uint8).contiguous()
    else:
        raise ValueError(f"Unexpected bits={bits}")

    # Store norms as fp16 bytes to minimize overhead.
    norms_fp16 = norms.to(dtype=torch.float16)  # [N,1]
    norm_bytes = norms_fp16.view(torch.uint8).view(-1, 2)  # [N,2]

    packed = torch.empty((x_flat.shape[0], packed_bytes), dtype=torch.uint8, device=x.device)
    packed[:, : indices_packed.shape[1]] = indices_packed.reshape(x_flat.shape[0], -1)
    packed[:, indices_packed.shape[1] : indices_packed.shape[1] + 2] = norm_bytes
    _debug_dump_packed(
        "quantize_to_packed_bytes(out)",
        packed,
        head_size=head_size,
        bits=bits,
    )
    return packed.reshape(*x.shape[:-1], packed_bytes)


def turboquant_dequantize_from_packed_bytes(
    packed: torch.Tensor, *, head_size: int, dtype: torch.dtype, bits: int = 4
) -> torch.Tensor:
    """
    Dequantize vectors from packed uint8 bytes produced by
    `turboquant_quantize_to_packed_bytes`.
    """
    logger.debug(
        "TurboQuant dequantize: packed=%s packed_dtype=%s device=%s bits=%d head_size=%d out_dtype=%s",
        tuple(packed.shape),
        packed.dtype,
        packed.device,
        bits,
        head_size,
        dtype,
    )
    packed_bytes = turboquant_packed_bytes_per_vector(head_size, bits=bits)
    row_w = packed.shape[-1]
    if row_w < packed_bytes:
        raise ValueError(
            f"Unexpected packed last dim {row_w} (need >= {packed_bytes})."
        )
    packed_logical = packed[..., :packed_bytes]
    if row_w != packed_bytes:
        packed_logical = packed_logical.contiguous()
    _debug_dump_packed(
        "dequantize_from_packed_bytes(in)", packed_logical, head_size=head_size, bits=bits
    )

    device_str = str(packed.device)
    quantizer = _get_quantizer(head_size, bits, device_str)

    packed_flat = packed_logical.reshape(-1, packed_bytes)
    idx_len = head_size // 2 if bits == 4 else head_size
    idx_packed = packed_flat[:, :idx_len]
    norm_bytes = packed_flat[:, idx_len : idx_len + 2]

    if bits == 4:
        indices = unpack_uint4(idx_packed, head_size)
    else:
        indices = idx_packed.to(dtype=torch.uint8)
    norms_fp16 = norm_bytes.contiguous().view(torch.float16).view(-1, 1)
    norms = norms_fp16.to(dtype=torch.float32)

    x_hat = quantizer.dequantize(indices, norms).to(dtype=dtype)
    return x_hat.reshape(*packed.shape[:-1], head_size)


def turboquant_store_kv(
    *,
    key: torch.Tensor,  # [T, H, D]
    value: torch.Tensor,  # [T, H, D]
    key_cache: torch.Tensor,  # [B, BS, H, P] uint8
    value_cache: torch.Tensor,  # [B, BS, H, P] uint8
    slot_mapping: torch.Tensor,  # [T] int32/int64
    bits: int = 4,
    bits_key: int | None = None,
    bits_value: int | None = None,
) -> None:
    """Quantize K/V and scatter into the paged cache using slot_mapping."""
    if key.numel() == 0:
        return
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported key dtype for turboquant: {key.dtype}")
    if value.dtype != key.dtype:
        raise ValueError("Key/value dtypes must match for turboquant.")

    bk = bits if bits_key is None else bits_key
    bv = bits if bits_value is None else bits_value
    slot_w = key_cache.shape[-1]
    if slot_w != value_cache.shape[-1]:
        raise ValueError("TurboQuant key/value cache slot width must match.")

    T, H, D = key.shape
    logger.debug(
        "TurboQuant store_kv: T=%d H=%d D=%d cache_blocks=%d block_size=%d bits_k=%d bits_v=%d",
        T,
        H,
        D,
        key_cache.shape[0],
        key_cache.shape[1],
        bk,
        bv,
    )
    block_size = key_cache.shape[1]

    def _pad_last(p: torch.Tensor) -> torch.Tensor:
        w = p.shape[-1]
        if w == slot_w:
            return p
        if w > slot_w:
            raise ValueError(f"packed width {w} > cache slot {slot_w}")
        return torch.nn.functional.pad(p, (0, slot_w - w))

    packed_k = _pad_last(turboquant_quantize_to_packed_bytes(key, bits=bk))  # [T,H,P]
    packed_v = _pad_last(turboquant_quantize_to_packed_bytes(value, bits=bv))  # [T,H,P]
    if _turboquant_debug_enabled():
        _debug_dump_packed(
            "store_kv(packed_k sample)", packed_k, head_size=D, bits=bk
        )
        _debug_dump_packed(
            "store_kv(packed_v sample)", packed_v, head_size=D, bits=bv
        )

    slot = slot_mapping.to(torch.int64)
    valid = slot >= 0
    if not torch.any(valid):
        return
    slot = slot[valid]
    tok_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

    block_idx = torch.div(slot, block_size, rounding_mode="floor")
    block_off = slot - block_idx * block_size

    key_cache[block_idx, block_off] = packed_k[tok_idx]
    value_cache[block_idx, block_off] = packed_v[tok_idx]


def turboquant_decode_kv_cache(
    *,
    key_cache: torch.Tensor,  # [B, BS, H, P] uint8
    value_cache: torch.Tensor,  # [B, BS, H, P] uint8
    head_size: int,
    dtype: torch.dtype,
    bits: int = 4,
    bits_key: int | None = None,
    bits_value: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the full paged cache into float tensors for attention compute."""
    bk = bits if bits_key is None else bits_key
    bv = bits if bits_value is None else bits_value
    k = turboquant_dequantize_from_packed_bytes(
        key_cache, head_size=head_size, dtype=dtype, bits=bk
    )
    v = turboquant_dequantize_from_packed_bytes(
        value_cache, head_size=head_size, dtype=dtype, bits=bv
    )
    return k, v

