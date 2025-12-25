import argparse
import json
import os
from typing import List, Dict, Any
import torch
import numpy as np
from PIL import Image
import open_clip

def load_json_items(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def pick_text(x: Dict[str, Any]) -> str:
    for k in ("text","caption","utterance","prompt","sentence"):
        v = x.get(k)
        if isinstance(v, str):
            return v
    return str(x)

def pick_image_rel(x: Dict[str, Any]) -> str:
    for k in ("img","image","image_path","file_name","filename"):
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
    return torch.nn.functional.normalize(x, dim=-1)

def encode_openclip(device, texts, images, batch_size=64):
    import torch
    import open_clip

    device = torch.device(device) if not isinstance(device, torch.device) else device
    use_amp = (device.type == "cuda")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model = model.to(device)
    model.eval()

    # # ---- TEXT ----
    # with torch.no_grad():
    #     text_tokens = tokenizer(texts).to(device)
    #     with torch.cuda.amp.autocast(enabled=use_amp):
    #         t = model.encode_text(text_tokens)
    #     t = normalize(t).detach().float().cpu()

    # --- encode text in streaming batches ---
    t_feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = tokenizer(batch_texts).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                t = model.encode_text(text_tokens)
            t = normalize(t).detach().float().cpu()
            t_feats.append(t)

            del text_tokens, t
            if use_amp:
                torch.cuda.empty_cache()

    t = torch.cat(t_feats, dim=0)


    # ---- IMAGES ----
    img_feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_pixels = torch.stack([preprocess(im) for im in batch_imgs]).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                v = model.encode_image(batch_pixels)

            v = normalize(v).detach().float().cpu()
            img_feats.append(v)

            del batch_pixels, v
            if use_amp:
                torch.cuda.empty_cache()

    img = torch.cat(img_feats, dim=0)  # float32 CPU
    return {"text": t, "image": img}


def losf_safe_whiten_filter(img_all: torch.Tensor, idxs_safe: List[int], idxs_harm: List[int], beta: float, eps: float = 1e-4) -> torch.Tensor:
    if not idxs_safe or not idxs_harm:
        return img_all
    Xs = img_all[idxs_safe].float()
    Xh = img_all[idxs_harm].float()
    mu_s = Xs.mean(dim=0, keepdim=True)
    Xs_c = Xs - mu_s
    Cs = (Xs_c.T @ Xs_c) / max(1, Xs_c.shape[0])
    d = Cs.shape[0]
    Cs = Cs + eps * torch.eye(d, device=Cs.device, dtype=Cs.dtype)
    lam_s, Us = torch.linalg.eigh(Cs)
    lam_s = torch.clamp(lam_s, min=eps)
    inv_sqrt = Us @ torch.diag((lam_s + eps).pow(-0.5)) @ Us.T
    sqrt = Us @ torch.diag((lam_s + eps).pow(0.5)) @ Us.T
    Xh_c = Xh - Xh.mean(dim=0, keepdim=True)
    Ch = (Xh_c.T @ Xh_c) / max(1, Xh_c.shape[0])
    Ch = Ch + eps * torch.eye(d, device=Ch.device, dtype=Ch.dtype)
    W = inv_sqrt
    Lam = W @ Ch @ W.T
    lam, U = torch.linalg.eigh(Lam)
    lam = torch.clamp(lam, min=eps)
    D = torch.diag(beta / (lam + beta))
    T = sqrt @ U @ D @ U.T @ W
    X_all_c = img_all - mu_s
    Zp = (X_all_c @ T.T) + mu_s
    return normalize(Zp)

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

def semantic_shift(text_base: torch.Tensor, text_post: torch.Tensor, labels: List[int]) -> Dict[str, float]:
    labels_np = np.array(labels)
    safe_mask = labels_np == 0
    unsafe_mask = labels_np == 1
    neutral_centroid = normalize(text_base[safe_mask].mean(dim=0, keepdim=True)) if safe_mask.any() else None
    unsafe_centroid = normalize(text_base[unsafe_mask].mean(dim=0, keepdim=True)) if unsafe_mask.any() else None
    res = {"decline_to_unsafe": None, "increase_to_neutral": None}
    if unsafe_centroid is not None:
        base_u = (text_base @ unsafe_centroid.T).squeeze(1)
        post_u = (text_post @ unsafe_centroid.T).squeeze(1)
        res["decline_to_unsafe"] = float((base_u - post_u).mean().cpu().item())
    if neutral_centroid is not None:
        base_n = (text_base @ neutral_centroid.T).squeeze(1)
        post_n = (text_post @ neutral_centroid.T).squeeze(1)
        res["increase_to_neutral"] = float((post_n - base_n).mean().cpu().item())
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=False, default="")
    parser.add_argument("--json_path", type=str, required=False, default="")
    parser.add_argument("--output_json", type=str, required=False, default="losf_metrics.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
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
    # base = encode_openclip(device, texts, images)
    base = encode_openclip(device, texts, images, batch_size=getattr(args, "batch_size", 64))

    n = len(items)
    idxs_safe = [i for i in range(n) if labels[i] == 0]
    idxs_harm = [i for i in range(n) if labels[i] == 1]
    post_image = losf_safe_whiten_filter(base["image"], idxs_safe, idxs_harm, args.beta)
    
    text_f = base["text"].float()
    img_f = base["image"].float()
    post_f = post_image.float()
    sim_base = similarity_matrix(text_f, img_f).cpu()
    sim_post = similarity_matrix(text_f, post_f).cpu()

    rec_base = recall_at_k(sim_base, args.k, labels)
    rec_post = recall_at_k(sim_post, args.k, labels)
    clipscore_safe_pre = diag_similarity(sim_base, labels, 0)
    clipscore_safe_post = diag_similarity(sim_post, labels, 0)
    shift = semantic_shift(base["text"], base["text"], labels)
    data = {
        "utility_recall@k_pre_S_V": rec_base["safe"],
        "utility_recall@k_post_S_V": rec_post["safe"],
        "harmful_recall@k_pre_U_V": rec_base["unsafe"],
        "harmful_recall@k_post_U_V": rec_post["unsafe"],
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
