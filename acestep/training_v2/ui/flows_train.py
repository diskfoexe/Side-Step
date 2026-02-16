"""
Wizard flow for training.

Training mode is auto-detected from the model variant (turbo vs base/sft).
Uses a step-list pattern for go-back navigation.  Step functions are
defined in ``flows_train_steps`` to keep this module under the LOC cap.

Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
"""

from __future__ import annotations

import argparse

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import GoBack, _esc, menu, step_indicator
from acestep.training_v2.ui.flows_common import build_train_namespace
from acestep.training_v2.ui.flows_train_steps import (
    step_config_mode,
    step_required,
    step_lora,
    step_lokr,
    step_training,
    step_cfg,
    step_logging,
    step_chunk_duration,
    step_advanced_device,
    step_advanced_optimizer,
    step_advanced_vram,
    step_advanced_training,
    step_advanced_dataloader,
    step_advanced_logging,
)


# ---- Step list builder ------------------------------------------------------

def _is_turbo_variant(answers: dict) -> bool:
    """Return ``True`` if the selected model is turbo or turbo-based.

    Detection is name-based only -- ``num_inference_steps`` is metadata
    and must not influence the training strategy.
    """
    base = answers.get("base_model", answers.get("model_variant", "turbo"))
    label_lower = base.lower() if isinstance(base, str) else ""
    if "turbo" in label_lower:
        return True
    if "base" in label_lower or "sft" in label_lower:
        return False
    # Unknown model name -- default to turbo (most common variant).
    return True


def _build_steps(answers: dict, config_mode: str, adapter_type: str = "lora") -> list[tuple[str, callable]]:
    """Return the ordered list of ``(label, step_fn)`` for this wizard run."""
    adapter_step = step_lokr if adapter_type == "lokr" else step_lora
    adapter_label = "LoKR Settings" if adapter_type == "lokr" else "LoRA Settings"

    steps = [
        ("Required Settings", step_required),
        (adapter_label, adapter_step),
        ("Training Settings", step_training),
    ]

    # CFG dropout settings only apply to base/sft (turbo doesn't use CFG)
    if not _is_turbo_variant(answers):
        steps.append(("CFG Dropout Settings", step_cfg))

    steps.append(("Logging & Checkpoints", step_logging))
    steps.append(("Latent Chunking", step_chunk_duration))

    if config_mode == "advanced":
        steps.extend([
            ("Device & Precision", step_advanced_device),
            ("Optimizer & Scheduler", step_advanced_optimizer),
            ("VRAM Savings", step_advanced_vram),
            ("Advanced Training", step_advanced_training),
            ("Data Loading", step_advanced_dataloader),
            ("Advanced Logging", step_advanced_logging),
        ])

    return steps


# ---- Preset helpers ---------------------------------------------------------

def _offer_load_preset(answers: dict) -> None:
    """Ask user if they want to load a preset; merge values into answers."""
    from acestep.training_v2.ui.presets import list_presets, load_preset

    presets = list_presets()
    if not presets:
        return

    options: list[tuple[str, str]] = [("fresh", "Start fresh (defaults)")]
    for p in presets:
        tag = " (built-in)" if p["builtin"] else ""
        desc = f" -- {p['description']}" if p["description"] else ""
        options.append((p["name"], f"{p['name']}{tag}{desc}"))

    choice = menu("Load a preset?", options, default=1, allow_back=True)

    if choice != "fresh":
        data = load_preset(choice)
        if data:
            answers.update(data)
            if is_rich_active() and console is not None:
                console.print(f"  [green]Loaded preset '{choice}'.[/]\n")
            else:
                print(f"  Loaded preset '{choice}'.\n")


def _offer_save_preset(answers: dict) -> None:
    """After wizard completes, offer to save settings as a preset.

    Errors from file I/O or name validation are caught and displayed
    so the user gets feedback rather than a silent failure.
    """
    from acestep.training_v2.ui.presets import save_preset
    from acestep.training_v2.ui.prompt_helpers import ask_bool, ask as _ask, section

    try:
        section("Save Preset")
        if not ask_bool("Save these settings as a reusable preset?", default=True):
            return
        name = _ask("Preset name", required=True)
        desc = _ask("Short description", default="")
        path = save_preset(name, desc, answers)

        # Verify the file was actually written
        if path.is_file():
            size = path.stat().st_size
            if is_rich_active() and console is not None:
                console.print(
                    f"  [green]Preset '{_esc(name)}' saved ({size} bytes)[/]\n"
                    f"  [dim]Location: {_esc(path)}[/]\n"
                )
            else:
                print(f"  Preset '{name}' saved ({size} bytes)")
                print(f"  Location: {path}\n")
        else:
            if is_rich_active() and console is not None:
                console.print(f"  [red]Warning: preset file not found after save: {_esc(path)}[/]\n")
            else:
                print(f"  Warning: preset file not found after save: {path}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        # Catch ValueError (bad name), OSError/PermissionError, etc.
        if is_rich_active() and console is not None:
            console.print(f"  [red]Failed to save preset: {_esc(exc)}[/]\n")
        else:
            print(f"  Failed to save preset: {exc}\n")


# ---- Public entry point -----------------------------------------------------

def _print_training_strategy(answers: dict) -> None:
    """Show the auto-detected training strategy after model selection."""
    if _is_turbo_variant(answers):
        msg = "Turbo detected -- using discrete 8-step sampling (no CFG)"
    else:
        msg = "Base/SFT detected -- using continuous sampling + CFG dropout"

    if is_rich_active() and console is not None:
        console.print(f"\n  [bold cyan]{msg}[/]\n")
    else:
        print(f"\n  {msg}\n")


def wizard_train(
    mode: str = "fixed",
    adapter_type: str = "lora",
    preset: dict | None = None,
) -> argparse.Namespace:
    """Interactive wizard for training.

    Training mode is always ``'fixed'``; turbo vs base/sft is
    auto-detected from the selected model variant.

    Args:
        mode: Training subcommand (always ``'fixed'``).
        adapter_type: Adapter type ('lora' or 'lokr').
        preset: Optional dict of pre-filled answer values (e.g. dataset_dir
            from the chain flow after preprocessing).  These values are
            used as defaults but do NOT suppress preset loading.

    Returns:
        A populated ``argparse.Namespace`` ready for dispatch.

    Raises:
        GoBack: If the user backs out of the very first step.
    """
    # Pre-fill any values the caller provided (e.g. dataset_dir from
    # the preprocess chain flow).  These are saved and restored after
    # preset loading so they always take priority.
    prefill: dict = dict(preset) if preset else {}
    answers: dict = dict(prefill)
    answers["adapter_type"] = adapter_type

    # Always offer to load a preset.  Pre-filled values (like
    # dataset_dir) are not in PRESET_FIELDS, so they survive the
    # update.  adapter_type is guarded below regardless.
    try:
        _offer_load_preset(answers)
    except GoBack:
        raise

    # Guard: adapter_type always comes from the menu selection, never
    # from a preset.  Pre-filled values also take priority over preset
    # values in case of overlap.
    answers["adapter_type"] = adapter_type
    answers.update(prefill)

    # Step 0: config depth
    try:
        step_config_mode(answers)
    except GoBack:
        raise

    config_mode = answers["config_mode"]
    steps = _build_steps(answers, config_mode, adapter_type)
    total = len(steps)
    i = 0
    printed_strategy = False

    while i < total:
        label, step_fn = steps[i]
        try:
            step_indicator(i + 1, total, label)
            step_fn(answers)
            i += 1

            # After model selection (step_required), show auto-detected
            # strategy and rebuild steps in case turbo/base changed.
            if step_fn is step_required and not printed_strategy:
                _print_training_strategy(answers)
                printed_strategy = True
                steps = _build_steps(answers, config_mode, adapter_type)
                total = len(steps)
        except GoBack:
            if i == 0:
                try:
                    step_config_mode(answers)
                except GoBack:
                    raise
                config_mode = answers["config_mode"]
                steps = _build_steps(answers, config_mode, adapter_type)
                total = len(steps)
                i = 0
                printed_strategy = False
            else:
                i -= 1

    _offer_save_preset(answers)

    return build_train_namespace(answers)
