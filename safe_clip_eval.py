import argparse
import json
import os
from typing import List, Dict, Any
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import open_clip

def load_json_items(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        if first.strip().startswith("{"):
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        else:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                items = data
            else:
                items = data.get("data", [])
    return items

def pick_text(x: Dict[str, Any]) -> str:
    for k in ("text","caption","utterance","prompt","sentence"):
        v = x.get(k)
        if isinstance(v, str):
            return v
    return str(x)

def pick_image_rel(x: Dict[str, Any]) -> str:
    for k in ("img","image","image_path","image_file","file_name","filename"):
        v = x.get(k)
        if isinstance(v, str):
            return v
    return ""

def pick_label(x: Dict[str, Any]) -> int:
    for k in ("label","is_hateful","unsafe","nsfw"):
        if k in x:
            v = x[k]
            if isinstance(v, bool):
                return int(v)
            try:
                return int(v)
            except Exception:
                return 0
    return 0

def load_images(image_dir: str, rel_paths: List[str]) -> List[Image.Image]:
    imgs = []
    for rp in rel_paths:
        p = rp
        if image_dir and rp and not os.path.isabs(rp):
            p = os.path.join(image_dir, rp)
        imgs.append(Image.open(p).convert("RGB"))
    return imgs

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True)

def encode_safeclip(model_id: str, device: torch.device, texts: List[str], images: List[Image.Image]) -> Dict[str, torch.Tensor]:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    with torch.no_grad():
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_outputs = model(**text_inputs)
        t = normalize(text_outputs.text_embeds)
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_outputs = model(**image_inputs)
        v = normalize(image_outputs.image_embeds)
    return {"text": t, "image": v}

def encode_openclip(device: torch.device, texts: List[str], images: List[Image.Image], model_name: str = "ViT-L/14", pretrained: str = "laion2b_s32b_b82k") -> Dict[str, torch.Tensor]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    with torch.no_grad():
        toks = tokenizer(texts)
        toks = toks.to(device)
        batch_pixels = torch.stack([preprocess(img) for img in images]).to(device)
        t = normalize(model.encode_text(toks))
        v = normalize(model.encode_image(batch_pixels))
    return {"text": t, "image": v}

def similarity_matrix(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return t @ v.T

def recall_at_k(sim: torch.Tensor, k: int, labels: List[int]) -> Dict[str, float]:
    n = sim.size(0)
    topk = torch.topk(sim, k=k, dim=1).indices
    hits = torch.arange(n, device=sim.device).unsqueeze(1)
    correct = (topk == hits).any(dim=1).cpu().numpy()
    labels_np = np.array(labels)
    safe_mask = labels_np == 0
    unsafe_mask = labels_np == 1
    safe_recall = float(correct[safe_mask].mean()) if safe_mask.any() else float("nan")
    unsafe_recall = float(correct[unsafe_mask].mean()) if unsafe_mask.any() else float("nan")
    return {"safe": safe_recall, "unsafe": unsafe_recall}

def diag_similarity(sim: torch.Tensor, labels: List[int], target_label: int) -> float:
    n = sim.size(0)
    diag = sim.diag()[:n].cpu().numpy()
    labels_np = np.array(labels)
    mask = labels_np == target_label
    return float(diag[mask].mean()) if mask.any() else float("nan")

def semantic_shift(text_base: torch.Tensor, text_safe: torch.Tensor, labels: List[int]) -> Dict[str, float]:
    labels_np = np.array(labels)
    safe_mask = labels_np == 0
    unsafe_mask = labels_np == 1
    neutral_centroid = normalize(text_base[safe_mask].mean(dim=0, keepdim=True)) if safe_mask.any() else None
    unsafe_centroid = normalize(text_base[unsafe_mask].mean(dim=0, keepdim=True)) if unsafe_mask.any() else None
    res = {"decline_to_unsafe": None, "increase_to_neutral": None}
    if unsafe_centroid is not None:
        base_u = (text_base @ unsafe_centroid.T).squeeze(1)
        safe_u = (text_safe @ unsafe_centroid.T).squeeze(1)
        res["decline_to_unsafe"] = float((base_u - safe_u).mean().cpu().item())
    if neutral_centroid is not None:
        base_n = (text_base @ neutral_centroid.T).squeeze(1)
        safe_n = (text_safe @ neutral_centroid.T).squeeze(1)
        res["increase_to_neutral"] = float((safe_n - base_n).mean().cpu().item())
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=False, default="")
    parser.add_argument("--json_path", type=str, required=False, default="")
    parser.add_argument("--output_json", type=str, required=False, default="safe_clip_metrics.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model_id", type=str, default="aimagelab/safeclip_vit-l_14")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.dry_run or not args.json_path or not os.path.exists(args.json_path):
        data = {
            "utility_recall@k_pre_S_V": None,
            "utility_recall@k_post_S_V": None,
            "harmful_recall@k_pre_U_V": None,
            "harmful_recall@k_post_U_V": None,
            "text_semantic_shift_decline_to_unsafe": None,
            "text_semantic_shift_increase_to_neutral": None,
            "clipscore_safe_pre": None,
            "clipscore_safe_post": None,
            "safety_rates_ASR": None,
            "safety_rates_USR": None
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return
    items = load_json_items(args.json_path)
    if args.limit and args.limit > 0:
        items = items[:args.limit]
    texts = [pick_text(x) for x in items]
    rels = [pick_image_rel(x) for x in items]
    labels = [pick_label(x) for x in items]
    images = load_images(args.image_dir, rels)
    safeclip = encode_safeclip(args.model_id, device, texts, images)
    base = encode_openclip(device, texts, images)
    sim_base = similarity_matrix(base["text"], base["image"]).cpu()
    sim_safe = similarity_matrix(safeclip["text"], safeclip["image"]).cpu()
    rec_base = recall_at_k(sim_base, args.k, labels)
    rec_safe = recall_at_k(sim_safe, args.k, labels)
    clipscore_safe_pre = diag_similarity(sim_base, labels, 0)
    clipscore_safe_post = diag_similarity(sim_safe, labels, 0)
    shift = semantic_shift(base["text"], safeclip["text"], labels)
    data = {
        "utility_recall@k_pre_S_V": rec_base["safe"],
        "utility_recall@k_post_S_V": rec_safe["safe"],
        "harmful_recall@k_pre_U_V": rec_base["unsafe"],
        "harmful_recall@k_post_U_V": rec_safe["unsafe"],
        "text_semantic_shift_decline_to_unsafe": shift["decline_to_unsafe"],
        "text_semantic_shift_increase_to_neutral": shift["increase_to_neutral"],
        "clipscore_safe_pre": clipscore_safe_pre,
        "clipscore_safe_post": clipscore_safe_post,
        "safety_rates_ASR": None,
        "safety_rates_USR": None
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()

