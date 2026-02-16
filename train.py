#!/usr/bin/env python3
"""
ACE-Step Training V2 -- CLI Entry Point

Usage:
    python train.py <subcommand> [args]

Subcommands:
    fixed            Train a LoRA/LoKR (auto-detects turbo vs base/sft)
    selective        Corrected training with dataset-specific module selection
    estimate         Gradient sensitivity analysis (no training)
    compare-configs  Compare module config JSON files
    convert          Convert a PEFT LoRA adapter to ComfyUI format

Examples:
    python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo \\
        --dataset-dir ./preprocessed_tensors/jazz --output-dir ./lora_output/jazz

    python train.py --help
"""

from __future__ import annotations

import gc
import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup (before any library imports that might configure logging)
# ---------------------------------------------------------------------------

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console (same as before)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)

# File (captures DEBUG+ including tracebacks)
# Guard against read-only working directories (e.g. some Windows setups)
try:
    _file_handler = logging.FileHandler("sidestep.log", mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_log_formatter)
    _log_handlers = [_console_handler, _file_handler]
except OSError:
    _log_handlers = [_console_handler]

logging.basicConfig(level=logging.DEBUG, handlers=_log_handlers)
logger = logging.getLogger("train")


def _has_subcommand() -> bool:
    """Check if sys.argv contains a recognized subcommand or --help."""
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        return True  # let argparse handle help
    known = {"vanilla", "fixed", "selective", "estimate", "compare-configs", "convert"}
    return bool(known & set(args))


def _cleanup_gpu() -> None:
    """Release GPU memory between session-loop iterations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _dispatch(args) -> int:
    """Route a parsed argparse.Namespace to the correct subcommand runner.

    Returns an int exit code (0 = success).
    """
    from acestep.training_v2.cli.common import validate_paths

    # -- Preprocessing (wizard flow) ----------------------------------------
    if getattr(args, "preprocess", False):
        return _run_preprocess(args)

    sub = args.subcommand

    # compare-configs and convert have their own validation
    if sub == "compare-configs":
        return _run_compare_configs(args)
    if sub == "convert":
        return _run_convert(args)

    # All other subcommands need path validation
    if not validate_paths(args):
        return 1

    if sub == "vanilla":
        print(
            "[INFO] Vanilla mode has been removed. Turbo models are now\n"
            "       automatically trained with discrete 8-step sampling.\n"
            "       Use 'fixed' instead -- it auto-detects turbo vs base/sft.",
            file=sys.stderr,
        )
        return 1

    elif sub == "fixed":
        from acestep.training_v2.cli.train_fixed import run_fixed
        return run_fixed(args)

    elif sub == "selective":
        return _run_selective(args)

    elif sub == "estimate":
        return _run_estimate(args)

    else:
        print(f"[FAIL] Unknown subcommand: {sub}", file=sys.stderr)
        return 1


def main() -> int:
    """Entry point for Side-Step training CLI.

    When invoked with a subcommand (``python train.py fixed ...``), runs
    that subcommand once and exits.  When invoked without arguments,
    launches the interactive wizard in a session loop so the user can
    preprocess, train, and manage presets without restarting.
    """
    # -- Compatibility check (non-fatal) ------------------------------------
    try:
        from acestep.training_v2._compat import check_compatibility
        check_compatibility()
    except Exception:
        pass  # never let the compat check itself crash the CLI

    # -- Direct CLI mode (subcommand given) ---------------------------------
    if _has_subcommand():
        from acestep.training_v2.settings import is_first_run
        if is_first_run():
            print(
                "[INFO] First-time setup not complete. "
                "Run 'python train.py' without arguments for the interactive setup wizard."
            )
        from acestep.training_v2.cli.common import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args()
        return _dispatch(args)

    # -- Interactive wizard session loop ------------------------------------
    from acestep.training_v2.ui.wizard import run_wizard_session

    last_code = 0
    for args in run_wizard_session():
        try:
            last_code = _dispatch(args)
        except Exception as exc:
            logger.exception("Unhandled error in session loop")
            print(f"[FAIL] {exc}", file=sys.stderr)
            last_code = 1
        finally:
            _cleanup_gpu()

    return last_code


# ===========================================================================
# Subcommand implementations
# ===========================================================================

def _run_preprocess(args) -> int:
    """Run the two-pass preprocessing pipeline."""
    from acestep.training_v2.preprocess import preprocess_audio_files

    audio_dir = getattr(args, "audio_dir", None)
    dataset_json = getattr(args, "dataset_json", None)
    tensor_output = getattr(args, "tensor_output", None)

    if not audio_dir and not dataset_json:
        print("[FAIL] --audio-dir or --dataset-json is required for preprocessing.", file=sys.stderr)
        return 1
    if not tensor_output:
        print("[FAIL] --tensor-output is required for preprocessing.", file=sys.stderr)
        return 1

    source_label = dataset_json if dataset_json else audio_dir

    # Show summary and confirm before starting
    print("\n" + "=" * 60)
    print("  Preprocessing Summary")
    print("=" * 60)
    print(f"  Source:        {source_label}")
    print(f"  Output:        {tensor_output}")
    print(f"  Checkpoint:    {args.checkpoint_dir}")
    print(f"  Model variant: {args.model_variant}")
    print(f"  Max duration:  {getattr(args, 'max_duration', 240.0)}s")
    print("=" * 60)
    print("[INFO] Two-pass pipeline (sequential model loading for low VRAM)")

    try:
        result = preprocess_audio_files(
            audio_dir=audio_dir,
            output_dir=tensor_output,
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            max_duration=getattr(args, "max_duration", 240.0),
            dataset_json=dataset_json,
            device=getattr(args, "device", "auto"),
            precision=getattr(args, "precision", "auto"),
        )
    except Exception as exc:
        print(f"[FAIL] Preprocessing failed: {exc}", file=sys.stderr)
        logger.exception("Preprocessing error")
        return 1
    finally:
        _cleanup_gpu()

    print(f"\n[OK] Preprocessing complete:")
    print(f"     Processed: {result['processed']}/{result['total']}")
    if result["failed"]:
        print(f"     Failed:    {result['failed']}")
    print(f"     Output:    {result['output_dir']}")
    print(f"\n[INFO] You can now train with:")
    print(f"       python train.py fixed --dataset-dir {result['output_dir']} ...")
    return 0


def _run_selective(args) -> int:
    """Run selective training (placeholder -- full implementation in Conversation C)."""
    print("[INFO] Selective training is not yet implemented.")
    print("[INFO] Use 'fixed' for corrected training, or 'estimate' for module analysis.")
    return 0


def _run_estimate(args) -> int:
    """Run gradient sensitivity estimation."""
    import json as _json
    from acestep.training_v2.estimate import run_estimation

    num_batches = getattr(args, "estimate_batches", 5) or 5

    # Show summary before starting
    print("\n" + "=" * 60)
    print("  Gradient Estimation Summary")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint_dir}")
    print(f"  Model variant: {args.model_variant}")
    print(f"  Dataset:       {args.dataset_dir}")
    print(f"  Batches:       {num_batches}")
    print(f"  Top-K:         {getattr(args, 'top_k', 16)}")
    print(f"  Granularity:   {getattr(args, 'granularity', 'module')}")
    print("=" * 60)
    print(f"[INFO] Running gradient estimation ({num_batches} batches) ...")
    try:
        results = run_estimation(
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            dataset_dir=args.dataset_dir,
            num_batches=num_batches,
            batch_size=args.batch_size,
            top_k=getattr(args, "top_k", 16) or 16,
            granularity=getattr(args, "granularity", "module") or "module",
        )
    except Exception as exc:
        print(f"[FAIL] Estimation failed: {exc}", file=sys.stderr)
        logger.exception("Estimation error")
        return 1
    finally:
        _cleanup_gpu()

    if not results:
        print("[WARN] No results -- dataset may be empty or model incompatible.")
        return 1

    # Display results
    print("\n" + "=" * 60)
    print(f"  Top-{len(results)} Modules by Gradient Sensitivity")
    print("=" * 60)
    for i, entry in enumerate(results, 1):
        print(f"  {i:3d}. {entry['module']:<50s}  {entry['sensitivity']:.6f}")
    print("=" * 60 + "\n")

    # Save to JSON
    output_path = getattr(args, "estimate_output", None) or "./estimate_results.json"
    try:
        with open(output_path, "w") as f:
            _json.dump(results, f, indent=2)
        print(f"[OK] Results saved to {output_path}")
    except OSError as exc:
        print(f"[WARN] Could not save results: {exc}", file=sys.stderr)

    return 0


def _run_compare_configs(args) -> int:
    """Compare module config JSON files (placeholder -- full implementation in Conversation D)."""
    from acestep.training_v2.cli.common import validate_paths
    if not validate_paths(args):
        return 1
    print("[INFO] compare-configs is not yet implemented.", file=sys.stderr)
    return 1


def _run_convert(args) -> int:
    """Convert a PEFT LoRA adapter to diffusers format for ComfyUI."""
    from acestep.training_v2.export_utils import convert_peft_to_diffusers

    adapter_dir = args.adapter_dir
    output = getattr(args, "output", None)

    print("\n" + "=" * 60)
    print("  Convert PEFT LoRA -> ComfyUI (diffusers format)")
    print("=" * 60)
    print(f"  Adapter dir: {adapter_dir}")
    if output:
        print(f"  Output:      {output}")
    else:
        print(f"  Output:      {adapter_dir}/pytorch_lora_weights.safetensors")
    print("=" * 60)

    try:
        out_path = convert_peft_to_diffusers(adapter_dir, output_path=output)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[FAIL] Conversion failed: {exc}", file=sys.stderr)
        logger.exception("Conversion error")
        return 1

    print(f"\n[OK] ComfyUI-compatible LoRA saved to: {out_path}")
    print("[INFO] Load this file in ComfyUI's LoRA loader node.")
    return 0


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
