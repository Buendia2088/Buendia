#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dap.py — Domain-Adaptive Pre-training (DAP) 多方法一键对比（含修复后的 Layer Expansion）
方法集合：
  • dapt_full           : 全量微调（new-only）
  • dapt_lora           : 冻结主干 + LoRA（new-only）
  • dapt_lora_replay    : LoRA + old_train 回放
  • dapt_lora_rewarm    : LoRA + cosine 调度 + warmup（lr re-warm & re-decay）
  • dapt_layer_exp      : 冻结主干 + 顶层残差 MLP（Layer Expansion，已修复：gated residual + zero-init + ln_out；
                           且默认不训练 lm_head；若显式开启 --expansion_tune_lm_head 将“先解绑”再训练）

输出：
  • {output_dir}/dap_summary.json / .csv
  • {output_dir}/dap_new_before_after.png / dap_old_before_after.png / dap_deltas.png（可用 --no_plot 关闭）

注意：
  • 训练过程关闭中途 safetensors（save_safetensors=False），避免 XLM-R 的 tied weights 报错
  • Expansion 自定义保存为 expansion.pt；其它模式正常 save_pretrained
"""

import os, math, argparse, json, random, csv
import warnings
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model

# ---------- 环境与告警 ----------
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # 静默 Windows symlink 提示
warnings.filterwarnings("once", category=FutureWarning)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- 数据 ----------
class LineByLineTextDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
    def __len__(self): return len(self.lines)
    def __getitem__(self, idx): return {"text": self.lines[idx]}

def load_text_dataset(path: str):
    ds = LineByLineTextDataset(path)
    return HFDataset.from_list([{"text": t["text"]} for t in ds])

def tokenize_dataset(raw_ds, tokenizer, seq_len):
    def _tok(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=seq_len,
            padding=False
        )
    return raw_ds.map(_tok, batched=True, remove_columns=["text"])


# ---------- 工具 ----------
def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False

def optionally_unfreeze_layernorm_and_bias(model, enable_ln=True, enable_bias=False):
    for name, p in model.named_parameters():
        if enable_ln and (".LayerNorm." in name or name.endswith(".layer_norm.weight") or name.endswith(".layer_norm.bias")):
            p.requires_grad = True
        if enable_bias and name.endswith(".bias"):
            p.requires_grad = True

def build_lora_model(model, target_modules=("query","value"), r=8, alpha=16, dropout=0.1):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules)
    )
    return get_peft_model(model, cfg)

def print_trainables(model, limit=0):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    all_params = sum(_p.numel() for _n, _p in model.named_parameters())
    ratio = 100.0 * trainable / max(1, all_params)
    print(f"Trainables: {len(names)} tensors | {trainable} / {all_params} ({ratio:.4f}%)")
    if limit > 0:
        for n in names[:limit]:
            print("  -", n)


# ---------- 修复后的 Layer Expansion ----------
class ResidualMLPLayer(nn.Module):
    """
    Gated Residual MLP：
      LN -> Linear(d, d*mult) -> GELU -> Dropout -> Linear(d*mult, d)[zero-init] -> Dropout -> gate * h + x
    关键：
      • fc2 zero-init + gate initial=0 → 初始“近似恒等”，稳定不炸
    """
    def __init__(self, d_model: int, mult: float = 2.0, p_drop: float = 0.1, init_zero: bool = True, gate_init: float = 0.0):
        super().__init__()
        hidden = int(d_model * mult)
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout2 = nn.Dropout(p_drop)
        self.gate = nn.Parameter(torch.tensor(gate_init))  # 学习型门控，初始 0

        if init_zero:
            nn.init.zeros_(self.fc2.weight)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h); h = self.act(h); h = self.dropout1(h)
        h = self.fc2(h); h = self.dropout2(h)
        return x + self.gate * h

class TopLayerExpander(nn.Module):
    """多个 ResidualMLPLayer + 输出 LayerNorm 稳定分布"""
    def __init__(self, d_model: int, num_layers: int, mult: float = 2.0, p_drop: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualMLPLayer(d_model, mult=mult, p_drop=p_drop, init_zero=True, gate_init=0.0)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        for lyr in self.layers:
            x = lyr(x)
        return self.ln_out(x)

class ExpandedForMaskedLM(nn.Module):
    """
    冻结底座 encoder，仅训练顶层扩展（以及可选“解绑后的”lm_head）
    roberta(**inputs) -> last_hidden_state -> expansion -> dropout -> lm_head -> logits
    """
    def __init__(self, base_mlm: AutoModelForMaskedLM, expansion: nn.Module | None,
                 tune_lm_head: bool = False, untie_lm_head: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.base = base_mlm
        self.expansion = expansion
        self.config = self.base.config
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.is_gradient_checkpointing = False

        # 冻结底座（除 lm_head 外）
        encoder = getattr(self.base, "roberta", None) or getattr(self.base, "bert", None) or None
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad = False
        else:
            for n, p in self.base.named_parameters():
                if not n.startswith("lm_head."):
                    p.requires_grad = False

        # 处理 lm_head：
        # 默认不训练（避免与 embeddings 绑权引发分布错位）
        # 若显式要求训练，则先“解绑”（clone 参数，打断共享），再开启 requires_grad
        for p in self.base.lm_head.parameters():
            p.requires_grad = False
        if tune_lm_head:
            # 仅在 untie=True 时执行解绑，保证不会影响 embeddings
            if untie_lm_head:
                dec = self.base.lm_head.decoder
                # 找到 embeddings
                embeddings = None
                if hasattr(self.base, "roberta"):
                    embeddings = self.base.roberta.embeddings.word_embeddings
                elif hasattr(self.base, "bert"):
                    embeddings = self.base.bert.embeddings.word_embeddings
                # 解绑权重/偏置
                if embeddings is not None:
                    dec.weight = nn.Parameter(embeddings.weight.detach().clone())
                else:
                    dec.weight = nn.Parameter(dec.weight.detach().clone())
                if hasattr(self.base.lm_head, "bias") and self.base.lm_head.bias is not None:
                    self.base.lm_head.bias = nn.Parameter(self.base.lm_head.bias.detach().clone())
            for p in self.base.lm_head.parameters():
                p.requires_grad = True

        # 开放 expansion 训练
        if self.expansion is not None:
            for p in self.expansion.parameters():
                p.requires_grad = True

    # 让 Trainer 兼容 GC
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True
        if hasattr(self.base, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            else:
                self.base.gradient_checkpointing_enable()
    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()
    def enable_input_require_grads(self):
        if hasattr(self.base, "enable_input_require_grads"):
            self.base.enable_input_require_grads()
        else:
            emb_fn = getattr(self.base, "get_input_embeddings", None)
            if callable(emb_fn):
                module = emb_fn()
                if module is not None:
                    module.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # 仅走 encoder，获得隐藏状态
        if hasattr(self.base, "roberta"):
            outputs = self.base.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        elif hasattr(self.base, "bert"):
            outputs = self.base.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        else:
            # 兜底：调用 base.forward，但只取 last_hidden_state
            outputs = self.base.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state  # (B, T, D)

        if self.expansion is not None:
            hidden = self.expansion(hidden, attention_mask=attention_mask)

        hidden = self.dropout(hidden)
        logits = self.base.lm_head(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


# ---------- 评测 ----------
@torch.no_grad()
def average_mlm_loss(model, dataset, data_collator, batch_size=16, num_workers=2):
    device = next(model.parameters()).device
    was_training = model.training
    reenable_gc = False
    if hasattr(model, "is_gradient_checkpointing") and getattr(model, "is_gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            reenable_gc = True

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=data_collator)
    total_loss, total_steps = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        total_loss += float(loss.detach().cpu()); total_steps += 1

    if reenable_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if was_training:
        model.train()
    if total_steps == 0:
        return None
    return total_loss / total_steps

def evaluate_loss(trainer: Trainer, eval_ds, data_collator, desc="eval"):
    metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=desc)
    key = f"{desc}_loss"
    loss = metrics.get(key, None)
    if loss is not None:
        ppl = math.exp(loss) if loss < 20 else float("inf")
        return {"loss": float(loss), "ppl_approx": float(ppl)}
    # 回退
    loss = average_mlm_loss(
        trainer.model, eval_ds, data_collator,
        batch_size=trainer.args.per_device_eval_batch_size,
        num_workers=trainer.args.dataloader_num_workers
    )
    if loss is None:
        return {"loss": None, "ppl_approx": None}
    ppl = math.exp(loss) if loss < 20 else float("inf")
    return {"loss": float(loss), "ppl_approx": float(ppl)}


# ---------- 稳健保存 ----------
def robust_save_model(trainer_model, tokenizer, out_dir, mode_tag, base_model_name_or_path, args):
    """
    - 扩展模式：保存 expansion.pt（含 expansion 与（若启用）解绑后的 lm_head）
    - 其它模式：save_pretrained；若不支持则保存 state_dict
    """
    save_dir = Path(out_dir) / mode_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    is_expansion = (trainer_model.__class__.__name__ == "ExpandedForMaskedLM") or \
                   (hasattr(trainer_model, "base") and hasattr(trainer_model, "expansion"))

    if is_expansion:
        exp = getattr(trainer_model, "expansion", None)
        base = getattr(trainer_model, "base", None)
        payload = {
            "base_model_name_or_path": base_model_name_or_path,
            "mode_tag": mode_tag,
            "expansion_state_dict": (exp.state_dict() if exp is not None else None),
            "lm_head_state_dict": (base.lm_head.state_dict() if (args.expansion_tune_lm_head) else None),
            "config": trainer_model.config.to_dict() if hasattr(trainer_model, "config") else {},
            "expansion_hparams": {
                "layers": args.expand_layers, "mult": args.expand_ffn_mult, "dropout": args.expansion_dropout
            }
        }
        torch.save(payload, save_dir / "expansion.pt")
        if base is not None and hasattr(base, "config"):
            base.config.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"[Saved Expansion] => {save_dir / 'expansion.pt'}")
    else:
        if hasattr(trainer_model, "save_pretrained"):
            trainer_model.save_pretrained(str(save_dir))
        else:
            torch.save(trainer_model.state_dict(), save_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(str(save_dir))
        print(f"[Saved Model] => {save_dir}")


# ---------- 单次训练 ----------
def train_dapt(tokenizer, base_model_name_or_path, train_ds, old_val, new_val,
               data_collator, args, out_dir, mode_tag,
               build_mode_fn, lr_scheduler_type="linear", warmup_ratio=0.0) -> Dict[str, Any]:

    print(f"\n=== DAP run: {mode_tag} ===")
    base = AutoModelForMaskedLM.from_pretrained(base_model_name_or_path)
    model = build_mode_fn(base, args)  # 构造 LoRA/Full/Expansion

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(Path(out_dir) / mode_tag),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        save_safetensors=False,   # 关键：避免 tied weights 报错
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": bool(args.gc_use_reentrant)},
        fp16=args.fp16,
        bf16=args.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    robust_save_model(trainer.model, tokenizer, out_dir, mode_tag, base_model_name_or_path, args)

    after_old = evaluate_loss(trainer, old_val, data_collator, desc=f"{mode_tag}_after_old")
    after_new = evaluate_loss(trainer, new_val, data_collator, desc=f"{mode_tag}_after_new")
    print(f"[After {mode_tag}] old_val: {after_old} | new_val: {after_new}")

    return {"after_old": after_old, "after_new": after_new}


# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser()
    # 数据与输出
    parser.add_argument("--old_train", type=str, default="data/old_train.txt")
    parser.add_argument("--old_val",   type=str, default="data/old_val.txt")
    parser.add_argument("--new_train", type=str, default="data/new_train.txt")
    parser.add_argument("--new_val",   type=str, default="data/new_val.txt")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--output_dir", type=str, default="outputs/dap_compare")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # LoRA 超参
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_targets", type=str, default="query,key,value")

    # 扩展（Layer Expansion）超参（安全默认：不训练 lm_head）
    parser.add_argument("--expand_layers", type=int, default=4)
    parser.add_argument("--expand_ffn_mult", type=float, default=2.0)
    parser.add_argument("--expansion_dropout", type=float, default=0.1)
    parser.add_argument("--expansion_tune_lm_head", action="store_true",
                        help="若显式开启，将先解绑 lm_head 再训练，以避免与 embeddings 的权重共享造成分布错位")

    # 回放
    parser.add_argument("--replay_ratio", type=float, default=0.0)

    # GC
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gc_use_reentrant", action="store_true")

    # 实验选择
    parser.add_argument("--experiments", type=str, default="dapt_lora,dapt_full,dapt_layer_exp,dapt_lora_replay,dapt_lora_rewarm",
                        help="逗号分隔：dapt_lora,dapt_full,dapt_layer_exp,dapt_lora_replay,dapt_lora_rewarm")
    parser.add_argument("--no_plot", action="store_true", help="不生成图片")

    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Loading tokenizer & base model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    try:
        tokenizer.clean_up_tokenization_spaces = True
    except Exception:
        pass

    # 加载数据
    def _load_and_tok(path): return tokenize_dataset(load_text_dataset(path), tokenizer, args.seq_len)
    old_train = _load_and_tok(args.old_train)
    old_val   = _load_and_tok(args.old_val)
    new_train = _load_and_tok(args.new_train)
    new_val   = _load_and_tok(args.new_val)

    print(f"Dataset sizes -> old_train:{len(old_train)} old_val:{len(old_val)} new_train:{len(new_train)} new_val:{len(new_val)}")

    # collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_prob)

    # 统一 before（使用纯底座）
    base_for_before = AutoModelForMaskedLM.from_pretrained(args.model_name)
    probe_args = TrainingArguments(
        output_dir=str(out_dir / "probe"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=2,
        logging_strategy="no",
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16 and torch.cuda.is_available(),
        eval_strategy="no",
        save_safetensors=False
    )
    trainer_probe = Trainer(
        model=base_for_before, args=probe_args,
        data_collator=data_collator, tokenizer=tokenizer
    )
    before_old = evaluate_loss(trainer_probe, old_val, data_collator, desc="before_old")
    before_new = evaluate_loss(trainer_probe, new_val, data_collator, desc="before_new")
    print(f"[Before DAP] old_val: {before_old} | new_val: {before_new}")

    # 构造器
    def build_lora(base, cfg):
        freeze_all_params(base)
        optionally_unfreeze_layernorm_and_bias(base, enable_ln=True, enable_bias=False)  # 如需更稳，可改为 enable_ln=False
        targets = tuple([t.strip() for t in cfg.lora_targets.split(",") if t.strip()])
        model = build_lora_model(base, target_modules=targets, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        model.print_trainable_parameters()
        return model

    def build_full(base, cfg):
        for p in base.parameters(): p.requires_grad = True
        print_trainables(base)
        return base

    def build_layer_exp(base, cfg):
        d_model = base.config.hidden_size
        expander = TopLayerExpander(d_model=d_model, num_layers=cfg.expand_layers,
                                    mult=cfg.expand_ffn_mult, p_drop=cfg.expansion_dropout)
        # 默认不训练 lm_head；若显式开启，将自动“解绑后”再训练
        model = ExpandedForMaskedLM(base, expander,
                                    tune_lm_head=cfg.expansion_tune_lm_head,
                                    untie_lm_head=True,
                                    dropout_p=cfg.expansion_dropout)
        print_trainables(model)
        return model

    # 训练集（new-only / replay）
    new_only = new_train
    if args.replay_ratio > 0:
        n_new = len(new_train)
        n_old = int(n_new * args.replay_ratio)
        old_sample = old_train.shuffle(seed=args.seed).select(range(min(n_old, len(old_train))))
        replay_mix = concatenate_datasets([new_train, old_sample]).shuffle(seed=args.seed)
    else:
        replay_mix = new_train

    # 计划的实验
    todo = [x.strip() for x in args.experiments.split(",") if x.strip()]
    results: Dict[str, Any] = {"before": {"old_val": before_old, "new_val": before_new}, "runs": {}}

    # dapt_lora
    if "dapt_lora" in todo:
        r = train_dapt(tokenizer, args.model_name, new_only, old_val, new_val, data_collator,
                       args, out_dir, "dapt_lora", build_lora, lr_scheduler_type="linear", warmup_ratio=0.0)
        results["runs"]["dapt_lora"] = r

    # dapt_full
    if "dapt_full" in todo:
        r = train_dapt(tokenizer, args.model_name, new_only, old_val, new_val, data_collator,
                       args, out_dir, "dapt_full", build_full, lr_scheduler_type="linear", warmup_ratio=0.0)
        results["runs"]["dapt_full"] = r

    # dapt_layer_exp（修复版）
    if "dapt_layer_exp" in todo:
        r = train_dapt(tokenizer, args.model_name, new_only, old_val, new_val, data_collator,
                       args, out_dir, "dapt_layer_exp", build_layer_exp,
                       lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio)  # 建议带 warmup 更稳
        results["runs"]["dapt_layer_exp"] = r

    # dapt_lora_replay
    if "dapt_lora_replay" in todo:
        r = train_dapt(tokenizer, args.model_name, replay_mix, old_val, new_val, data_collator,
                       args, out_dir, "dapt_lora_replay", build_lora, lr_scheduler_type="linear", warmup_ratio=0.0)
        results["runs"]["dapt_lora_replay"] = r

    # dapt_lora_rewarm（cosine + warmup）
    if "dapt_lora_rewarm" in todo:
        r = train_dapt(tokenizer, args.model_name, new_only, old_val, new_val, data_collator,
                       args, out_dir, "dapt_lora_rewarm", build_lora,
                       lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio)
        results["runs"]["dapt_lora_rewarm"] = r

    # 计算 Δ 指标
    def safe(x, k): 
        return None if x is None else (x.get(k, None) if isinstance(x, dict) else None)
    for name, r in results["runs"].items():
        after_old = safe(r["after_old"], "loss")
        after_new = safe(r["after_new"], "loss")
        b_old = safe(results["before"]["old_val"], "loss")
        b_new = safe(results["before"]["new_val"], "loss")
        r["delta_old"] = (after_old - b_old) if (after_old is not None and b_old is not None) else None  # 越小越好
        r["delta_new"] = (b_new - after_new) if (after_new is not None and b_new is not None) else None  # 越大越好

    # 写 JSON
    with open(out_dir / "dap_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"==> Wrote {out_dir / 'dap_summary.json'}")

    # 写 CSV
    rows: List[List[Any]] = [["run","old_before","old_after","delta_old","new_before","new_after","delta_new"]]
    for name, r in results["runs"].items():
        old_b = safe(results["before"]["old_val"], "loss")
        new_b = safe(results["before"]["new_val"], "loss")
        old_a = safe(r["after_old"], "loss")
        new_a = safe(r["after_new"], "loss")
        rows.append([name, old_b, old_a, r["delta_old"], new_b, new_a, r["delta_new"]])
    with open(out_dir / "dap_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(rows)
    print(f"==> Wrote {out_dir / 'dap_summary.csv'}")

    # 可选绘图
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            labels = list(results["runs"].keys())
            old_b = safe(results["before"]["old_val"], "loss")
            new_b = safe(results["before"]["new_val"], "loss")
            old_after = [safe(results["runs"][k]["after_old"], "loss") for k in labels]
            new_after = [safe(results["runs"][k]["after_new"], "loss") for k in labels]
            delta_old = [results["runs"][k]["delta_old"] for k in labels]
            delta_new = [results["runs"][k]["delta_new"] for k in labels]

            x = list(range(len(labels))); width = 0.35

            # new before/after
            plt.figure(figsize=(10,5))
            plt.title("New-domain loss: before vs after")
            plt.xticks(x, labels, rotation=15, ha="right")
            plt.bar([i - width/2 for i in x], [new_b]*len(x), width, label="before (new)")
            plt.bar([i + width/2 for i in x], new_after, width, label="after (new)")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / "dap_new_before_after.png", dpi=200); plt.close()

            # old before/after
            plt.figure(figsize=(10,5))
            plt.title("Old-domain loss: before vs after")
            plt.xticks(x, labels, rotation=15, ha="right")
            plt.bar([i - width/2 for i in x], [old_b]*len(x), width, label="before (old)")
            plt.bar([i + width/2 for i in x], old_after, width, label="after (old)")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / "dap_old_before_after.png", dpi=200); plt.close()

            # deltas
            plt.figure(figsize=(10,5))
            plt.title("Deltas: Δold=after-before (↓ good), Δnew=before-after (↑ good)")
            plt.xticks(x, labels, rotation=15, ha="right")
            plt.bar([i - width/2 for i in x], delta_old, width, label="Δ old")
            plt.bar([i + width/2 for i in x], delta_new, width, label="Δ new")
            plt.axhline(0, linestyle="--")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / "dap_deltas.png", dpi=200); plt.close()

            print(f"==> Plots saved to {out_dir}")
        except Exception as e:
            print("Plot skipped due to:", repr(e))


if __name__ == "__main__":
    main()
