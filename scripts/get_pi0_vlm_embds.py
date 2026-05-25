"""
Extract Pi0 VLM embeddings for each task/instruction pair in ref_exp.json.

Two weight sets are supported in a single run:
  1. Base pretrained PaliGemma weights (no Bridge fine-tuning) — always extracted.
  2. A post-trained Pi0 .pt checkpoint — extracted when --checkpoint_path is given.

Output files per task:
  {task_name}_{paligemma_model_name}_pretrained_embds.npz
  {task_name}_{checkpoint_stem}_finetuned_embds.npz   (only when checkpoint provided)
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.vla.pizero import PiZero
from src.model.vla.processing import VLAProcessor

PALIGEMMA_REPO_ID = "google/paligemma-3b-pt-224"
_DEFAULT_PRETRAINED_PATH = os.path.join(
    os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")),
    "paligemma-3b-pt-224",
)


def ensure_paligemma_weights(path: str) -> str:
    """Return path after ensuring PaliGemma safetensors weights are present.

    Downloads from HuggingFace if the directory is missing or contains no
    .safetensors files.  Requires huggingface_hub (already a transformers dep).
    """
    has_weights = os.path.isdir(path) and any(
        f.endswith(".safetensors") for f in os.listdir(path)
    )
    if has_weights:
        print(f"Found existing PaliGemma weights at {path}")
        return path

    print(f"PaliGemma weights not found at {path}")
    print(f"Downloading {PALIGEMMA_REPO_ID} from HuggingFace Hub → {path}")
    os.makedirs(path, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=PALIGEMMA_REPO_ID, local_dir=path)
    print("Download complete.")
    return path


def load_json_strip_comments(path: str) -> dict:
    """Parse JSON after stripping JS-style // comments."""
    with open(path) as f:
        content = f.read()
    content = re.sub(r"//[^\n]*", "", content)
    return json.loads(content)


def load_image_tensor(image_path: str, size: int = 224) -> torch.Tensor:
    """Return uint8 image tensor [1, 3, H, W]."""
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
    return torch.as_tensor(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


def extract_embeddings(
    model: PiZero,
    processor: VLAProcessor,
    image_tensor: torch.Tensor,
    instruction: str,
    config,
    device: torch.device,
    dtype: torch.dtype,
):
    """Run a single image+instruction pair through Pi0's VLM and return embeddings."""
    model_inputs = processor(text=[instruction], images=image_tensor)
    input_ids = model_inputs["input_ids"].to(device)
    pixel_values = model_inputs["pixel_values"].to(dtype).to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    causal_mask, vlm_position_ids, proprio_position_ids, _ = (
        model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
    )
    image_text_proprio_mask, _ = model.split_full_mask_into_submasks(causal_mask)

    # Zero proprio — we care only about image+language VLM representation
    proprios = torch.zeros(
        1, config.cond_steps, config.proprio_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        all_layers_emb, last_layer_emb = model.get_vlm_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_text_proprio_mask=image_text_proprio_mask.to(device),
            vlm_position_ids=vlm_position_ids.to(device),
            proprio_position_ids=proprio_position_ids.to(device),
            proprios=proprios,
        )

    return (
        all_layers_emb.float().cpu().numpy(),
        last_layer_emb.float().cpu().numpy(),
    )


def load_checkpoint(model: PiZero, checkpoint_path: str) -> None:
    """Load a post-trained Pi0 .pt checkpoint into model in-place."""
    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    # strip "_orig_mod." prefix that torch.compile adds when saving
    data["model"] = {
        k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
    }
    model.load_state_dict(data["model"], strict=True)


def run_extraction(
    model: PiZero,
    processor: VLAProcessor,
    tasks: dict,
    image_base_dir: str,
    config,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: str,
    file_suffix: str,
) -> None:
    """Extract embeddings for every task+instruction and save one .npz per task."""
    for task_name, task_data in tasks.items():
        print(f"\n[Task] {task_name}")

        image_path = os.path.join(image_base_dir, task_data["image_path"])
        image_tensor = load_image_tensor(image_path)

        # base instruction first, then all variants
        instructions = {"base": task_data["base_command"]}
        instructions.update(task_data["variants"])

        npz_data = {}
        for inst_key, instruction in instructions.items():
            print(f"  [{inst_key}] '{instruction}'")
            all_layers, last_layer = extract_embeddings(
                model, processor, image_tensor, instruction, config, device, dtype
            )
            npz_data[f"{inst_key}_all_layers"] = all_layers  # [1, heads*head_dim]
            npz_data[f"{inst_key}_last_layer"] = last_layer  # [1, heads*head_dim]
            print(f"         embedding shape: {all_layers.shape}")

        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        out_path = os.path.join(task_dir, f"{task_name}_{file_suffix}.npz")
        np.savez_compressed(out_path, **npz_data)
        print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Pi0 VLM embeddings (pretrained and/or finetuned) for ref_exp tasks."
    )
    parser.add_argument(
        "--ref_exp_path",
        default=os.path.join(os.path.dirname(__file__), "../../media/ref_exp.jsonc"),
        help="Path to ref_exp.json",
    )
    parser.add_argument(
        "--image_base_dir",
        default=os.path.join(os.path.dirname(__file__), "../../media"),
        help="Directory that image_path entries in ref_exp.json are relative to",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "../../embeddings/pi0"),
        help="Where to write .npz embedding files (task subdirs created automatically)",
    )
    parser.add_argument(
        "--pretrained_model_path",
        default=_DEFAULT_PRETRAINED_PATH,
        help=(
            f"Path to base paligemma-3b-pt-224 checkpoint directory "
            f"(default: {_DEFAULT_PRETRAINED_PATH}). "
            "Downloaded automatically from HuggingFace if not present."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to a post-trained Pi0 .pt checkpoint.  When provided, finetuned "
             "embeddings are extracted in addition to pretrained ones.",
    )
    parser.add_argument(
        "--config_path",
        default=os.path.join(os.path.dirname(__file__), "../config/train/bridge.yaml"),
        help="Path to bridge.yaml config",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help="Subset of task names to process (default: all tasks in ref_exp). "
             "E.g. --tasks put_carrot_on_plate put_knife_on_plate",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 (recommended for GPU with bf16 support)",
    )
    args = parser.parse_args()

    pretrained_model_path = ensure_paligemma_weights(args.pretrained_model_path)

    # Load config and override pretrained path to avoid needing TRANSFORMERS_CACHE
    config = OmegaConf.load(args.config_path)
    config.pretrained_model_path = pretrained_model_path

    paligemma_name = os.path.basename(pretrained_model_path.rstrip("/"))
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    print(f"PaliGemma base: {paligemma_name}")
    print(f"Device: {device}  Dtype: {dtype}")
    if args.checkpoint_path:
        print(f"Post-trained checkpoint: {args.checkpoint_path}")

    # Build model architecture (shared across both weight sets)
    print("\nBuilding Pi0 model...")
    model = PiZero(config)
    model.tie_action_proprio_weights()
    model.to(dtype).to(device)
    model.eval()

    # Tokenizer + processor (same for both weight sets)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path, padding_side="right"
    )
    processor = VLAProcessor(
        tokenizer,
        num_image_tokens=config.vision.config.num_image_tokens,
        max_seq_len=config.max_seq_len,
    )

    tasks = load_json_strip_comments(args.ref_exp_path)
    if args.tasks:
        unknown = set(args.tasks) - set(tasks)
        if unknown:
            raise ValueError(f"Unknown task(s): {unknown}. Available: {list(tasks)}")
        tasks = {k: tasks[k] for k in args.tasks}
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: base pretrained PaliGemma weights (no Bridge fine-tuning) ──
    print("\n=== Extracting PRETRAINED embeddings ===")
    model.load_pretrained_weights()
    model.freeze_all_weights()
    run_extraction(
        model, processor, tasks,
        args.image_base_dir, config, device, dtype,
        args.output_dir,
        file_suffix=f"{paligemma_name}_pretrained_embds",
    )

    # ── Phase 2: post-trained checkpoint weights (optional) ──
    if args.checkpoint_path:
        checkpoint_stem = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        print(f"\n=== Extracting FINETUNED embeddings  ({checkpoint_stem}) ===")
        load_checkpoint(model, args.checkpoint_path)
        run_extraction(
            model, processor, tasks,
            args.image_base_dir, config, device, dtype,
            args.output_dir,
            file_suffix=f"{checkpoint_stem}_finetuned_embds",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
