"""
TCM-VL 图文检索（对齐/识别）Baseline

目标：先用“闭集（closed-set）”把检索评测闭环跑通：
- 输入：图文配对数据（jsonl）
- 输出：Image→Text / Text→Image 的 Recall@K、MRR

说明：
- 无框（no box）场景非常适合用图→文检索来定义“识别”。
- 这里支持一个 concept_id 对应多条文本（别名/拉丁名/描述），评测时把同 concept_id 的所有文本都视为正样本。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class PairItem:
    image_path: str
    text: str
    concept_id: str


def _read_jsonl(path: str) -> List[PairItem]:
    items: List[PairItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                PairItem(
                    image_path=str(obj["image"]),
                    text=str(obj["text"]),
                    concept_id=str(obj["concept_id"]),
                )
            )
    if not items:
        raise ValueError(f"空数据：{path}")
    return items


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _unique_images(items: List[PairItem]) -> List[Tuple[str, str]]:
    """
    返回：[(image_path, concept_id)]，按 image_path 去重。
    """
    seen: Dict[str, str] = {}
    for it in items:
        # 如果同一张图出现多个 concept_id，说明数据有歧义；这里保留首次出现的 concept_id。
        if it.image_path not in seen:
            seen[it.image_path] = it.concept_id
    return list(seen.items())


def _build_text_candidates(items: List[PairItem]) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """
    返回：
    - texts: 候选文本列表
    - concept_ids: 与 texts 对应的 concept_id 列表
    - concept_to_text_indices: concept_id -> 该 concept 的所有文本索引
    """
    texts: List[str] = []
    concept_ids: List[str] = []
    concept_to_text_indices: Dict[str, List[int]] = {}
    for it in items:
        idx = len(texts)
        texts.append(it.text)
        concept_ids.append(it.concept_id)
        concept_to_text_indices.setdefault(it.concept_id, []).append(idx)
    return texts, concept_ids, concept_to_text_indices


@torch.no_grad()
def _encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts"):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
        feats = feats.float().cpu().numpy()
        vecs.append(feats)
    return _l2norm(np.concatenate(vecs, axis=0))


@torch.no_grad()
def _encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encode images"):
        batch_paths = image_paths[i : i + batch_size]
        images: List[Image.Image] = []
        for p in batch_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"找不到图片：{p}")
            images.append(Image.open(p).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = feats.float().cpu().numpy()
        vecs.append(feats)
    return _l2norm(np.concatenate(vecs, axis=0))


def _recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def _mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / (ranks + 1)))


def _eval_i2t(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    img_concepts: List[str],
    concept_to_text_indices: Dict[str, List[int]],
) -> Dict[str, float]:
    sims = img_emb @ txt_emb.T  # [N_img, N_txt] cosine similarity
    ranks: List[int] = []
    for i, cid in enumerate(img_concepts):
        gt = concept_to_text_indices.get(cid, [])
        if not gt:
            continue
        order = np.argsort(-sims[i])  # desc
        # 取该图像命中的“最好名次”
        best_rank = int(np.min([np.where(order == j)[0][0] for j in gt]))
        ranks.append(best_rank)
    if not ranks:
        raise ValueError("评测失败：没有可用的正样本（concept_id 没有匹配到候选文本）")
    ranks_arr = np.asarray(ranks, dtype=np.int64)
    return {
        "i2t_R@1": _recall_at_k(ranks_arr, 1),
        "i2t_R@5": _recall_at_k(ranks_arr, 5),
        "i2t_R@10": _recall_at_k(ranks_arr, 10),
        "i2t_MRR": _mrr(ranks_arr),
    }


def _eval_t2i(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    img_concepts: List[str],
    txt_concepts: List[str],
) -> Dict[str, float]:
    sims = txt_emb @ img_emb.T  # [N_txt, N_img]
    # 为每条文本，正样本是所有同 concept_id 的图像。这里把“最好名次”作为该文本的 rank。
    concept_to_img_indices: Dict[str, List[int]] = {}
    for i, cid in enumerate(img_concepts):
        concept_to_img_indices.setdefault(cid, []).append(i)

    ranks: List[int] = []
    for t_idx, cid in enumerate(txt_concepts):
        gt = concept_to_img_indices.get(cid, [])
        if not gt:
            continue
        order = np.argsort(-sims[t_idx])
        best_rank = int(np.min([np.where(order == j)[0][0] for j in gt]))
        ranks.append(best_rank)
    if not ranks:
        raise ValueError("评测失败：没有可用的正样本（concept_id 没有匹配到候选图像）")
    ranks_arr = np.asarray(ranks, dtype=np.int64)
    return {
        "t2i_R@1": _recall_at_k(ranks_arr, 1),
        "t2i_R@5": _recall_at_k(ranks_arr, 5),
        "t2i_R@10": _recall_at_k(ranks_arr, 10),
        "t2i_MRR": _mrr(ranks_arr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/example_retrieval.jsonl", help="jsonl，字段：image/text/concept_id")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch16", help="HuggingFace 模型名")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="retrieval_metrics.json", help="输出指标 json 文件")
    args = parser.parse_args()

    device = torch.device(args.device)
    items = _read_jsonl(args.data)

    # 闭集：候选文本库固定为 items 里出现过的所有 text（可包含别名/描述等）
    texts, txt_concepts, concept_to_text_indices = _build_text_candidates(items)
    unique_imgs = _unique_images(items)
    img_paths = [p for (p, _) in unique_imgs]
    img_concepts = [cid for (_, cid) in unique_imgs]

    print(f"[INFO] unique images: {len(img_paths)}")
    print(f"[INFO] text candidates: {len(texts)}")
    print(f"[INFO] concepts: {len(set(txt_concepts))}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] device: {device}")

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()

    txt_emb = _encode_texts(model, processor, texts, device, batch_size=args.batch_size)
    img_emb = _encode_images(model, processor, img_paths, device, batch_size=args.batch_size)

    metrics = {}
    metrics.update(_eval_i2t(img_emb, txt_emb, img_concepts, concept_to_text_indices))
    metrics.update(_eval_t2i(img_emb, txt_emb, img_concepts, txt_concepts))

    print("\n===== Retrieval Metrics (closed-set) =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] saved metrics to: {args.out}")


if __name__ == "__main__":
    main()


