# -*- coding: utf-8 -*-
import os
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.cache_utils import StaticCache


# ----------------------------
# Utils
# ----------------------------
def ensure_local_alpaca(data_dir: str) -> str:
    """
    确保 Alpaca 数据存在于本地，只在第一次下载；之后一直使用本地文件。
    返回本地 json 文件路径。
    """
    os.makedirs(data_dir, exist_ok=True)
    local_json = os.path.join(data_dir, "alpaca_data_cleaned.json")
    if os.path.isfile(local_json):
        print(f"[Data] 使用本地数据: {local_json}")
        return local_json

    print("[Data] 本地未找到，正在从 Hugging Face 下载 yahma/alpaca-cleaned ...")
    ds = load_dataset("yahma/alpaca-cleaned")
    # 原始字段：instruction, input, output
    # 直接保存完整训练集为一个 json 文件
    records = [dict(r) for r in ds["train"]]
    with open(local_json, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Data] 已保存到本地: {local_json}")
    return local_json


def build_prompt(example: Dict) -> str:
    """
    构造简洁 SFT 文本：不使用 chat template（兼容所有 tokenizer），
    与之前逻辑一致。
    """
    instr = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()

    if inp:
        return f"Instruction:\n{instr}\n\nInput:\n{inp}\n\nAnswer:\n{out}"
    else:
        return f"Instruction:\n{instr}\n\nAnswer:\n{out}"


# ----------------------------
# Dataset
# ----------------------------
class AlpacaTxtDataset(Dataset):
    def __init__(self, tokenizer, data_path: str, train_size: int, eval_size: int, max_length: int, split: str):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取本地 jsonl
        rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        # 随机划分：先打乱，再切片
        rng = random.Random(42)
        rng.shuffle(rows)
        if split == "train":
            rows = rows[:train_size]
        else:
            rows = rows[train_size:train_size + eval_size]

        self.prompts = [build_prompt(r) for r in rows]

        # 直接在 __init__ 里做 tokenize（简单稳定）
        self.inputs = tokenizer(
            self.prompts,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # causal lm 监督，labels = input_ids（不需要额外 shift，Trainer 内做）
        self.labels = []
        for ids in self.inputs["input_ids"]:
            self.labels.append(ids.copy())

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


# ----------------------------
# Prefix Tuning (StaticCache 版本)
# ----------------------------
class StaticPrefixForCausalLM(nn.Module):
    """
    将 Prefix Tuning 适配到 HuggingFace 新的 KV Cache API：
    - 显式构建 StaticCache，并在 forward 前把 prefix K/V 写进去
    - 向基座 forward 传入 past_key_values=cache 与 cache_position
    - 只训练 prefix 参数；冻结基座
    """
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        num_virtual_tokens: int = 30,
        prefix_hidden_size: int = 512,
        init_std: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model

        cfg = base_model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        # Qwen2 使用 GQA：k/v head 数可能小于 attention heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)

        self.num_virtual_tokens = num_virtual_tokens

        # prefix token 的可学习 embedding（P, H）
        self.prefix_embeddings = nn.Parameter(
            torch.empty(num_virtual_tokens, self.hidden_size)
        )
        nn.init.normal_(self.prefix_embeddings, mean=0.0, std=init_std)

        # MLP：将 hidden_size 投影到 2 * L * (kv_heads * head_dim)
        # 2 表示 K/V
        out_dim = 2 * self.num_layers * (self.num_kv_heads * self.head_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, prefix_hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(prefix_hidden_size, out_dim, bias=True),
            nn.Dropout(dropout),
        )

        # 冻结基座，只训练 prefix
        self.base_model.requires_grad_(False)
        self.prefix_embeddings.requires_grad_(True)
        self.proj.requires_grad_(True)

    def _materialize_prefix_kv(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        生成每一层的 prefix K/V，返回两个 list，长度=num_layers，
        每个元素形状为 [B, num_kv_heads, P, head_dim]。
        """
        P = self.num_virtual_tokens
        L = self.num_layers
        Hkv = self.num_kv_heads
        Dh = self.head_dim

        # (P, H) -> (P, 2*L*Hkv*Dh)
        proj = self.proj(self.prefix_embeddings.to(device))
        proj = proj.view(P, L, 2, Hkv, Dh)  # (P, L, 2, Hkv, Dh)
        proj = proj.permute(1, 2, 0, 3, 4).contiguous()  # (L, 2, P, Hkv, Dh)

        k_list, v_list = [], []
        for l in range(L):
            k_l = proj[l, 0]  # (P, Hkv, Dh)
            v_l = proj[l, 1]  # (P, Hkv, Dh)
            # 变换为 [B, Hkv, P, Dh]
            k_l = k_l.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            v_l = v_l.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            k_list.append(k_l.to(dtype=dtype))
            v_list.append(v_l.to(dtype=dtype))
        return k_list, v_list

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        device = input_ids.device
        batch_size, S = input_ids.shape
        P = self.num_virtual_tokens

        # 生成 prefix K/V
        k_list, v_list = self._materialize_prefix_kv(
            batch_size=batch_size, device=device, dtype=self.base_model.dtype
        )

        # 构建 StaticCache，cache_len = prefix 长度 + 本次序列长度
        cache_len = P + S
        cache = StaticCache(
            config=self.base_model.config,
            max_batch_size=batch_size,
            max_cache_len=cache_len,
            device=device,
            dtype=self.base_model.dtype,
        )

        # 先把 prefix 写入到 cache 的 [0..P-1] 位置
        pos_prefix = torch.arange(P, device=device, dtype=torch.long)
        for layer_idx in range(self.num_layers):
            cache.update(
                k_list[layer_idx],
                v_list[layer_idx],
                layer_idx=layer_idx,
                cache_kwargs={"cache_position": pos_prefix},
            )

        # 真实 token 的 cache_position 为 [P .. P+S-1]
        cache_position = torch.arange(P, P + S, device=device, dtype=torch.long)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        return outputs


# ----------------------------
# Training / Eval
# ----------------------------
def compute_metrics(eval_preds):
    # 因为这里是语言建模，简单返回 ppl 与 avg loss
    (losses, _), _ = eval_preds  # Trainer 的 predict 返回 (loss, logits)，此处 logits 不用
    # 兼容性：有时 losses 可能是标量
    try:
        mean_loss = float(sum(losses) / len(losses))
    except TypeError:
        mean_loss = float(losses)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"eval_loss": mean_loss, "eval_ppl": ppl}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--method", type=str, choices=["prefix"], default="prefix")
    p.add_argument("--num_virtual_tokens", type=int, default=30)
    p.add_argument("--prefix_hidden_size", type=int, default=512)
    p.add_argument("--data_dir", type=str, default="./data/alpaca")
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--eval_size", type=int, default=200)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--out", type=str, default="./runs/alpaca_prefix")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 确保本地数据
    local_json = ensure_local_alpaca(args.data_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    train_ds = AlpacaTxtDataset(tokenizer, local_json, args.train_size, args.eval_size, args.max_length, split="train")
    eval_ds = AlpacaTxtDataset(tokenizer, local_json, args.train_size, args.eval_size, args.max_length, split="eval")

    # Model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=None,
    )
    base.to("cuda" if torch.cuda.is_available() else "cpu")

    assert args.method == "prefix", "此版本仅实现 Prefix Tuning 的可靠跑通。"

    model = StaticPrefixForCausalLM(
        base_model=base,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_hidden_size=args.prefix_hidden_size,
        init_std=0.02,
        dropout=0.0,
    )

    # 训练设置：remove_unused_columns=False 以免 Trainer 过滤掉必要字段
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=False,            # Prefix 依赖 use_cache=True，避免与 ckpt 冲突
        remove_unused_columns=False,
        dataloader_pin_memory=False,             # Windows 环境更稳
        report_to=[],
    )

    # 计时：体现算力开销（秒）
    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    t1 = time.time()

    # 训练后，做一次 eval
    eval_metrics = trainer.evaluate()
    t2 = time.time()

    # 统计信息（保存到 out_dir）
    stats = {
        "method": "prefix",
        "model": args.model,
        "num_virtual_tokens": args.num_virtual_tokens,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "train_runtime_s": round(t1 - t0, 3),
        "eval_runtime_s": round(t2 - t1, 3),
        "final_train_loss": float(train_output.training_loss) if hasattr(train_output, "training_loss") else None,
        "eval_metrics": eval_metrics,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "all_params": sum(p.numel() for p in model.parameters()),
        "trainable_ratio_%": round(
            100.0 * sum(p.numel() for p in model.parameters() if p.requires_grad)
            / max(1, sum(p.numel() for p in model.parameters())),
            4,
        ),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_stats_prefix.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n==== Train/Eval Summary ====")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
