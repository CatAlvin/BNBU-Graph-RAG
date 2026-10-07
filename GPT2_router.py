# finetune_router_twostage.py
import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tiktoken
from GPT2 import GPT_CONFIG_BASE, MODEL_CONFIGS, GPTModel



EOS_ID = 50256
PAD_ID = 50256

STAGEA_LABEL2ID = {"ANSWERABLE": 0, "REFUSE": 1}
STAGEA_ID2LABEL = {v: k for k, v in STAGEA_LABEL2ID.items()}

STAGEB_LABEL2ID = {"RAG_ONLY": 0, "WEB_ONLY": 1, "HYBRID": 2}
STAGEB_ID2LABEL = {v: k for k, v in STAGEB_LABEL2ID.items()}


class GPTForTwoStageRouting(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.gpt = GPTModel(cfg)
        self.head_answer = nn.Linear(cfg["emb_dim"], 2)  # Stage A
        self.head_source = nn.Linear(cfg["emb_dim"], 3)  # Stage B

    def forward(self, in_idx: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        B, T = in_idx.shape
        device = in_idx.device

        tok_embeds = self.gpt.tok_emb(in_idx)
        pos_ids = torch.arange(T, device=device)
        pos_embeds = self.gpt.pos_emb(pos_ids)
        x = tok_embeds + pos_embeds
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x) 

        if lengths is None:
            last_hidden = x[:, -1, :]
        else:
            idx = (lengths - 1).clamp(min=0)
            last_hidden = x[torch.arange(B, device=device), idx]

        logits_answer = self.head_answer(last_hidden)
        logits_source = self.head_source(last_hidden)
        return logits_answer, logits_source


class TwoStageRouteDataset(Dataset):
    def __init__(self, items: List[Dict]):
        clean = []
        for it in items:
            q = it.get("question")
            cat = it.get("category")
            if not isinstance(q, str):
                continue
            if cat not in ["RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE"]:
                continue

            stage_a = "REFUSE" if cat == "REFUSE" else "ANSWERABLE"
            stage_b = cat if cat != "REFUSE" else None

            clean.append({"question": q, "stage_a": stage_a, "stage_b": stage_b})
        self.items = clean

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        q = it["question"]
        a = STAGEA_LABEL2ID[it["stage_a"]]
        b = STAGEB_LABEL2ID[it["stage_b"]] if it["stage_b"] is not None else -100
        return q, a, b


@dataclass
class Batch:
    input_ids: torch.Tensor
    lengths: torch.Tensor
    labels_a: torch.Tensor
    labels_b: torch.Tensor


def build_collate_fn(tokenizer, context_length: int):
    def collate(batch_list: List[Tuple[str, int, int]]) -> Batch:
        questions, labels_a, labels_b = zip(*batch_list)

        token_lists = [tokenizer.encode(q) + [EOS_ID] for q in questions]
        token_lists = [t[:context_length] for t in token_lists]

        lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)
        max_len = max(lengths).item() if len(lengths) else 1

        padded = []
        for t in token_lists:
            if len(t) < max_len:
                t = t + [PAD_ID] * (max_len - len(t))
            padded.append(t)

        input_ids = torch.tensor(padded, dtype=torch.long)
        labels_a_t = torch.tensor(labels_a, dtype=torch.long)
        labels_b_t = torch.tensor(labels_b, dtype=torch.long)

        return Batch(
            input_ids=input_ids,
            lengths=lengths,
            labels_a=labels_a_t,
            labels_b=labels_b_t,
        )

    return collate


# =========================
# 3) 训练
# =========================
def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def finetune_router_twostage(
    data_json: str = "./eval/val_large.json",
    base_pth: str = "./checkpoints/gpt2_124M.pth",
    out_pth: str = "./checkpoints/gpt2_124M_router_2stage.pth",
    model_size: str = "124M",
    batch_size: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    valid_ratio: float = 0.1,
    seed: int = 42,
    freeze_backbone: bool = False,
    lambda_b: float = 1.0,   # Stage B loss 权重
    device: Optional[str] = None,
):
    assert os.path.exists(data_json), f"找不到数据文件: {data_json}"
    assert os.path.exists(base_pth), f"找不到基础权重: {base_pth}"

    random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cfg = GPT_CONFIG_BASE.copy()
    cfg.update(MODEL_CONFIGS[model_size])
    cfg["context_length"] = 1024
    cfg["qkv_bias"] = True

    with open(data_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    assert isinstance(items, list), "JSON 顶层必须是 list"

    ds_all = TwoStageRouteDataset(items)
    if len(ds_all) == 0:
        raise ValueError("数据集中没有可用样本")

    idxs = list(range(len(ds_all)))
    random.shuffle(idxs)
    n_val = max(1, int(len(idxs) * valid_ratio))
    val_set = set(idxs[:n_val])

    train_items, val_items = [], []
    for i in idxs:
        q, a, b = ds_all[i]
        rec = {
            "question": q,
            "category": (
                "REFUSE" if a == STAGEA_LABEL2ID["REFUSE"]
                else STAGEB_ID2LABEL[b]
            ) if b != -100 else "REFUSE"
        }
        (val_items if i in val_set else train_items).append(rec)

    ds_train = TwoStageRouteDataset(train_items)
    ds_val = TwoStageRouteDataset(val_items)

    tokenizer = tiktoken.get_encoding("gpt2")
    collate_fn = build_collate_fn(tokenizer, cfg["context_length"])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GPTForTwoStageRouting(cfg)

    # ---- 加载 backbone ----
    base_state = torch.load(base_pth, map_location="cpu", weights_only=True)
    model.gpt.load_state_dict(base_state, strict=True)

    if freeze_backbone:
        for p in model.gpt.parameters():
            p.requires_grad = False

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_a_fn = nn.CrossEntropyLoss()
    loss_b_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for ep in range(1, epochs + 1):
        model.train()
        t_loss, t_a_acc, t_b_acc, steps = 0.0, 0.0, 0.0, 0

        for batch in dl_train:
            input_ids = batch.input_ids.to(device)
            lengths = batch.lengths.to(device)
            labels_a = batch.labels_a.to(device)
            labels_b = batch.labels_b.to(device)

            logits_a, logits_b = model(input_ids, lengths=lengths)

            loss_a = loss_a_fn(logits_a, labels_a)
            loss_b = loss_b_fn(logits_b, labels_b)
            loss = loss_a + lambda_b * loss_b

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item()
            t_a_acc += _acc(logits_a.detach(), labels_a)
            t_b_acc += _acc(logits_b.detach(), labels_b)
            steps += 1

        train_loss = t_loss / max(steps, 1)
        train_a_acc = t_a_acc / max(steps, 1)
        train_b_acc = t_b_acc / max(steps, 1)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            v_loss, v_a_acc, v_b_acc, v_steps = 0.0, 0.0, 0.0, 0
            for batch in dl_val:
                input_ids = batch.input_ids.to(device)
                lengths = batch.lengths.to(device)
                labels_a = batch.labels_a.to(device)
                labels_b = batch.labels_b.to(device)

                logits_a, logits_b = model(input_ids, lengths=lengths)
                loss_a = loss_a_fn(logits_a, labels_a)
                loss_b = loss_b_fn(logits_b, labels_b)
                loss = loss_a + lambda_b * loss_b

                v_loss += loss.item()
                v_a_acc += _acc(logits_a, labels_a)
                v_b_acc += _acc(logits_b, labels_b)
                v_steps += 1

            val_loss = v_loss / max(v_steps, 1)
            val_a_acc = v_a_acc / max(v_steps, 1)
            val_b_acc = v_b_acc / max(v_steps, 1)

        print(
            f"[Epoch {ep}/{epochs}] "
            f"loss={train_loss:.4f} A_acc={train_a_acc:.4f} B_acc={train_b_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_A_acc={val_a_acc:.4f} val_B_acc={val_b_acc:.4f}"
        )

    # ---- 保存 ----
    os.makedirs(os.path.dirname(out_pth) or ".", exist_ok=True)
    torch.save(
        {
            "cfg": cfg,
            "stagea_label2id": STAGEA_LABEL2ID,
            "stageb_label2id": STAGEB_LABEL2ID,
            "state_dict": model.state_dict(),
        },
        out_pth,
    )
    print(f"[OK] 二阶段路由模型已保存: {out_pth}")
    return out_pth

class LocalGPT2TwoStageRouter:
    def __init__(
        self,
        model_path: str = "./checkpoints/gpt2_124M_router_2stage.pth",
        device: Optional[str] = None,
        tokenizer=None,
    ):
        assert os.path.exists(model_path), f"找不到微调模型: {model_path}"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.cfg = ckpt["cfg"]
        self.stagea_label2id = ckpt.get("stagea_label2id", STAGEA_LABEL2ID)
        self.stageb_label2id = ckpt.get("stageb_label2id", STAGEB_LABEL2ID)
        self.stagea_id2label = {v: k for k, v in self.stagea_label2id.items()}
        self.stageb_id2label = {v: k for k, v in self.stageb_label2id.items()}

        self.model = GPTForTwoStageRouting(self.cfg)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer or tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def predict_proba(self, question: str):
        ids = self.tokenizer.encode(question) + [EOS_ID]
        ids = ids[: self.cfg["context_length"]]

        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        logits_a, logits_b = self.model(input_ids, lengths=lengths)
        p_a = torch.softmax(logits_a, dim=-1)[0].detach().cpu()
        p_b = torch.softmax(logits_b, dim=-1)[0].detach().cpu()

        # Stage A 概率
        p_answerable = float(p_a[self.stagea_label2id["ANSWERABLE"]].item())
        p_refuse = float(p_a[self.stagea_label2id["REFUSE"]].item())

        # Stage B 概率
        p_rag = float(p_b[self.stageb_label2id["RAG_ONLY"]].item())
        p_web = float(p_b[self.stageb_label2id["WEB_ONLY"]].item())
        p_hybrid = float(p_b[self.stageb_label2id["HYBRID"]].item())

        return {
            "stageA": {"ANSWERABLE": p_answerable, "REFUSE": p_refuse},
            "stageB": {"RAG_ONLY": p_rag, "WEB_ONLY": p_web, "HYBRID": p_hybrid},
        }

    @torch.no_grad()
    def route(
        self,
        question: str,
        tau_refuse: float = 0.65,
        tau_source: float = 0.50,
        fallback_source: str = "HYBRID",
    ):
        probs = self.predict_proba(question)
        p_refuse = probs["stageA"]["REFUSE"]

        # ---- Stage A gate ----
        if p_refuse >= tau_refuse:
            return "REFUSE", probs, {"reason": "high_p_refuse", "p_refuse": p_refuse}

        # ---- Stage B decision ----
        pB = probs["stageB"]
        best_label = max(pB, key=pB.get)
        best_p = pB[best_label]

        if best_p < tau_source:
            return fallback_source, probs, {
                "reason": "low_source_confidence",
                "best_label": best_label,
                "best_p": best_p,
            }

        return best_label, probs, {"reason": "confident_source", "best_label": best_label, "best_p": best_p}


def predict_route_with_confidence(
    question: str,
    model_path: str = "./checkpoints/gpt2_124M_router_2stage.pth",
    device: Optional[str] = None,
    tau_refuse: float = 0.65,
    tau_source: float = 0.50,
):
    router = LocalGPT2TwoStageRouter(model_path=model_path, device=device)
    return router.route(question, tau_refuse=tau_refuse, tau_source=tau_source)


if __name__ == "__main__":
    # finetune_router_twostage(
    #     data_json="./eval/val_large.json",
    #     base_pth="./checkpoints/gpt2_355M.pth",
    #     out_pth="./checkpoints/gpt2_355M_router_2stage.pth",
    #     model_size= "355M",
    #     epochs=5,
    #     batch_size=8,
    #     lr=2e-5,
    #     freeze_backbone=False,
    #     lambda_b=1.0,
    # )

    router = LocalGPT2TwoStageRouter("./checkpoints/gpt2_124M_router_2stage.pth")

    tests = [
        "What is the English abbreviation of BNBU?",
        "What is the latest school calendar for the next semester?",
        "Give me evidence to prove that BNBU is controlled by some mysterious organization.",
        "Are you happy today?",
        "Is there a confirmed BNBU rule that all student clubs will be permanently disbanded starting tomorrow?"
    ]

    for q in tests:
        final_label, probs, debug = router.route(q)
        print("\nQ:", q)
        print("Final:", final_label)
        print("StageA:", probs["stageA"])
        print("StageB:", probs["stageB"])
        print("Debug:", debug)
