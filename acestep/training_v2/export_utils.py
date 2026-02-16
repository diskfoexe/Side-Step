"""Convert PEFT LoRA adapters to diffusers format for ComfyUI compatibility.

PEFT saves adapter weights with a ``base_model.model.`` key prefix and an
adapter-name segment (e.g. ``.lora_A.default.weight``).  ComfyUI and other
diffusers-based tools expect keys *without* that prefix and without the
adapter name (e.g. ``.lora_A.weight``).

This module provides a single public function, ``convert_peft_to_diffusers``,
that reads a PEFT adapter directory and writes a
``pytorch_lora_weights.safetensors`` file usable by ComfyUI.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Keys saved by PEFT always start with this prefix.
_PEFT_PREFIX = "base_model.model."

# PEFT v0.5+ inserts the adapter name between lora_{A,B} and .weight.
# e.g.  .lora_A.default.weight  ->  .lora_A.weight
_ADAPTER_NAME_RE = re.compile(r"\.lora_([AB])\.[^.]+\.weight$")


def _strip_peft_key(key: str) -> str:
    """Convert a single PEFT state-dict key to diffusers format.

    1. Remove the ``base_model.model.`` prefix (applied once).
    2. Remove the adapter-name segment from ``lora_{A,B}.<name>.weight``.
    """
    new_key = key
    if new_key.startswith(_PEFT_PREFIX):
        new_key = new_key[len(_PEFT_PREFIX):]
    new_key = _ADAPTER_NAME_RE.sub(r".lora_\1.weight", new_key)
    return new_key


def _validate_adapter_dir(adapter_dir: str) -> Tuple[str, Optional[str]]:
    """Return (safetensors_path, config_path_or_None) or raise."""
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(
            f"No adapter_model.safetensors found in {adapter_dir}. "
            "Is this a PEFT LoRA adapter directory?"
        )
    cfg = config_path if os.path.isfile(config_path) else None
    if cfg is None:
        logger.warning(
            "adapter_config.json not found in %s -- the PEFT adapter "
            "may be incomplete, but conversion will proceed.",
            adapter_dir,
        )
    return safetensors_path, cfg


def convert_peft_to_diffusers(
    adapter_dir: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert a PEFT LoRA adapter to diffusers format for ComfyUI.

    Args:
        adapter_dir: Path to a PEFT adapter directory containing
            ``adapter_model.safetensors`` (and optionally
            ``adapter_config.json``).
        output_path: Destination file path.  Defaults to
            ``pytorch_lora_weights.safetensors`` inside *adapter_dir*.

    Returns:
        The absolute path of the written diffusers safetensors file.

    Raises:
        FileNotFoundError: If *adapter_dir* or the safetensors file
            does not exist.
        RuntimeError: If conversion produces an empty state dict.
    """
    from safetensors.torch import load_file, save_file

    safetensors_path, _ = _validate_adapter_dir(adapter_dir)

    if output_path is None:
        output_path = os.path.join(adapter_dir, "pytorch_lora_weights.safetensors")

    peft_sd: Dict = load_file(safetensors_path)

    if not peft_sd:
        raise RuntimeError(
            f"adapter_model.safetensors in {adapter_dir} is empty -- "
            "nothing to convert."
        )

    diffusers_sd: Dict = {}
    converted_count = 0
    unchanged_count = 0

    for key, tensor in peft_sd.items():
        new_key = _strip_peft_key(key)
        diffusers_sd[new_key] = tensor
        if new_key != key:
            converted_count += 1
        else:
            unchanged_count += 1

    if not diffusers_sd:
        raise RuntimeError("Conversion produced an empty state dict.")

    # Warn if nothing was actually changed (keys already clean).
    if converted_count == 0:
        logger.warning(
            "No keys had the PEFT prefix -- the adapter may already be "
            "in diffusers format.  Writing output anyway."
        )

    save_file(diffusers_sd, output_path)

    logger.info(
        "Converted %d keys (%d already clean) -> %s",
        converted_count,
        unchanged_count,
        output_path,
    )
    return os.path.abspath(output_path)
