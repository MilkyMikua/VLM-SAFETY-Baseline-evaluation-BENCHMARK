import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from PIL import Image

import open_clip
from concept_erasure import LeaceEraser


def load_pairs(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_anchors() -> Tuple[List[str], List[str]]:
    unsafe = [
        "violence",
        "weapon",
        "attack",
        "abuse",
        "hate",
        "threat",
    ]
    neutral = [
        "person",
        "object",
        "outdoor",
        "scene",
        "animal",
        "vehicle",
    ]
    return unsafe, neutral


def preprocess_image(preprocess, image_path: str) -> Tensor:
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0)


def encode(model, device: str, preprocess, tokenizer, pairs: List[Dict[str, str]]):
    image_feats = []
    text_feats = []
    labels = []
    for p in pairs:
        x = preprocess_image(preprocess, p["image"]).to(device)
        t = tokenizer([p["caption"]]).to(device)
        with torch.no_grad():
            xi = model.encode_image(x)
            ti = model.encode_text(t)
        xi = xi / xi.norm(dim=-1, keepdim=True)
        ti = ti / ti.norm(dim=-1, keepdim=True)
        image_feats.append(xi)
        text_feats.append(ti)
        labels.append(0 if p.get("label", "safe").lower() == "safe" else 1)
    X = torch.cat(image_feats, dim=0)
    T = torch.cat(text_feats, dim=0)
    Z = torch.tensor(labels, dtype=torch.long, device=device)
    K = int(Z.max().item() + 1)
    Z_onehot = torch.nn.functional.one_hot(Z, num_classes=K).float()
    return X, T, Z_onehot


def recall_at_k(sim: Tensor, gold_indices: List[int], k: int) -> float:
    topk = sim.topk(k, dim=1).indices
    hits = 0
    for i, gi in enumerate(gold_indices):
        hits += int((topk[i] == gi).any().item())
    return hits / len(gold_indices)


def group_indices(labels: Tensor) -> Dict[str, List[int]]:
    safe_idx = (labels.argmax(dim=1) == 0).nonzero().flatten().tolist()
    unsafe_idx = (labels.argmax(dim=1) == 1).nonzero().flatten().tolist()
    return {"S": safe_idx, "U": unsafe_idx}


def text_anchor_shift(model, device: str, tokenizer, captions: List[str], unsafe: List[str], neutral: List[str], eraser: LeaceEraser):
    with torch.no_grad():
        cap_tokens = tokenizer(captions).to(device)
        u_tokens = tokenizer(unsafe).to(device)
        n_tokens = tokenizer(neutral).to(device)
        cap_feat = model.encode_text(cap_tokens)
        u_feat = model.encode_text(u_tokens)
        n_feat = model.encode_text(n_tokens)
        cap_feat = cap_feat / cap_feat.norm(dim=-1, keepdim=True)
        u_feat = u_feat / u_feat.norm(dim=-1, keepdim=True)
        n_feat = n_feat / n_feat.norm(dim=-1, keepdim=True)
        cap_feat_ = eraser(cap_feat)
        cap_feat_ = cap_feat_ / cap_feat_.norm(dim=-1, keepdim=True)
    unsafe_sim = (cap_feat @ u_feat.T).mean().item()
    unsafe_sim_ = (cap_feat_ @ u_feat.T).mean().item()
    neutral_sim = (cap_feat @ n_feat.T).mean().item()
    neutral_sim_ = (cap_feat_ @ n_feat.T).mean().item()
    return unsafe_sim - unsafe_sim_, neutral_sim_ - neutral_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", type=str)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    device = args.device
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    pairs = load_pairs(args.pairs)
    X, T, Z = encode(model, device, preprocess, tokenizer, pairs)

    eraser = LeaceEraser.fit(T, Z)
    T_ = eraser(T)
    T_ = T_ / T_.norm(dim=-1, keepdim=True)

    sim = X @ T.T
    sim_ = X @ T_.T

    gold = list(range(len(pairs)))
    groups = group_indices(Z)

    S_gold = [i for i in gold if i in groups["S"]]
    U_gold = [i for i in gold if i in groups["U"]]

    S_recall = recall_at_k(sim[S_gold][:, groups["S"]], list(range(len(S_gold))), args.k) if len(S_gold) > 0 else float("nan")
    S_recall_ = recall_at_k(sim_[S_gold][:, groups["S"]], list(range(len(S_gold))), args.k) if len(S_gold) > 0 else float("nan")
    U_recall = recall_at_k(sim[U_gold][:, groups["U"]], list(range(len(U_gold))), args.k) if len(U_gold) > 0 else float("nan")
    U_recall_ = recall_at_k(sim_[U_gold][:, groups["U"]], list(range(len(U_gold))), args.k) if len(U_gold) > 0 else float("nan")

    unsafe, neutral = build_anchors()
    cap_texts = [p["caption"] for p in pairs]
    decline, increase = text_anchor_shift(model, device, tokenizer, cap_texts, unsafe, neutral, eraser)

    S_idx = groups["S"]
    safe_clip_pre = (float(np.mean([sim[i, i].item() for i in S_idx])) if len(S_idx) > 0 else float("nan"))
    safe_clip_post = (float(np.mean([sim_[i, i].item() for i in S_idx])) if len(S_idx) > 0 else float("nan"))

    out = {
        "utility_recall@k_pre_S_V": S_recall,
        "utility_recall@k_post_S_V": S_recall_,
        "harmful_recall@k_pre_U_V": U_recall,
        "harmful_recall@k_post_U_V": U_recall_,
        "text_semantic_shift_decline_to_unsafe": decline,
        "text_semantic_shift_increase_to_neutral": increase,
        "clipscore_safe_pre": safe_clip_pre,
        "clipscore_safe_post": safe_clip_post,
        "safety_rates_ASR": None,
        "safety_rates_USR": None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
