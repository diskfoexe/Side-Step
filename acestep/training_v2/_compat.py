"""
ACE-Step Compatibility Check for Side-Step.

Side-Step is a sidecar that imports from the upstream ``acestep.training``
package.  This module records which ACE-Step revision Side-Step was tested
against and provides a lightweight check that warns (non-fatally) when the
host environment may have diverged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version pin
# ---------------------------------------------------------------------------

TESTED_ACESTEP_COMMIT = "46116a6"
"""Short SHA of the upstream ``ace-step/ACE-Step-1.5`` commit that this
version of Side-Step was tested and verified against."""

SIDESTEP_VERSION = "0.3.2-beta"
"""Current Side-Step release string."""


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def check_compatibility() -> None:
    """Verify that critical upstream symbols exist.

    This is intentionally non-fatal: it prints a warning and continues
    if something looks off.  The goal is to give the user a heads-up
    before a mysterious ``ImportError`` deep in the training loop.
    """
    warnings: list[str] = []

    # 1. Can we import the data module?
    try:
        from acestep.training.data_module import PreprocessedDataModule  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import acestep.training.data_module.PreprocessedDataModule"
        )

    # 2. Can we import LoRA utilities?
    try:
        from acestep.training.lora_utils import inject_lora_into_dit  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import acestep.training.lora_utils.inject_lora_into_dit"
        )

    # 3. Can we import the upstream trainer?
    try:
        from acestep.training.trainer import LoRATrainer  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import acestep.training.trainer.LoRATrainer"
        )

    # 4. Can we import upstream configs?
    try:
        from acestep.training.configs import TrainingConfig  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import acestep.training.configs.TrainingConfig"
        )

    if warnings:
        msg = (
            f"[Side-Step] Compatibility warning (tested against ACE-Step "
            f"commit {TESTED_ACESTEP_COMMIT}):\n"
        )
        for w in warnings:
            msg += f"  - {w}\n"
        msg += (
            "  Side-Step may still work, but if you hit import errors,\n"
            "  ensure you are running inside a compatible ACE-Step-1.5 checkout."
        )
        logger.warning(msg)
        print(f"\n{msg}\n")
    else:
        logger.debug(
            "[Side-Step] Compatibility check passed (pin: %s)",
            TESTED_ACESTEP_COMMIT,
        )
