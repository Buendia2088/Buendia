#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — CPT + 参数扩展 (Block/Layer) / LoRA / No-LoRA / Full Finetune
修复点（按你要求的 1、2 两种方法）：
  1) 在 TrainingArguments 中关闭中途 safetensors 保存（save_safetensors=False），避免 tied weights 报错
  2) 保存阶段采用“模式感知”的稳健保存：Expansion 用 torch.save(payload)；其它模式仍走 save_pretrained

仍保留：
  • --compare_rewarm（无 warmup vs cosine+warmup 对比）
  • 稳健评测回退（Trainer.evaluate 无值时手写平均 MLM loss）
  • 回放混采（concatenate_datasets）
  • GC 修复（gradient_checkpointing + enable_input_require_grads）
  • LoRA / No-LoRA / Full finetune 模式
"""

import os, math, argparse, json, random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model


# ---------------------- 环境与告警控制 ----------------------
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # 静默 Windows symlink 提示
warnings.filterwarnings("once", category=FutureWarning)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------- 数据集 ----------------------
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


# ---------------------- LoRA/冻结工具 ----------------------
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

def print_trainables(model, limit=40):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    all_params = sum(_p.numel() for _n, _p in model.named_parameters())  # 修复变量名
    print(f"Trainable tensors: {len(names)} | trainable params: {trainable} || all params: {all_params} || trainable%: {trainable/all_params*100:.4f}")
    for n in names[:limit]:
        print("  -", n)
    if len(names) > limit:
        print(f"  ... (+{len(names)-limit} more)")


# ---------------------- Block/Layer Expansion 实现 ----------------------
class ResidualMLPLayer(nn.Module):
    """轻量残差 MLP：LN -> Linear -> GELU -> Dropout -> Linear -> Dropout + Residual"""
    def __init__(self, d_model: int, mult: float = 2.0, p_drop: float = 0.1):
        super().__init__()
        hidden = int(d_model * mult)
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout1(h)
        h = self.fc2(h)
        h = self.dropout2(h)
        return x + h  # 残差


class TopBlockExpander(nn.Module):
    """若干 TransformerEncoder 层叠加在底座输出之上（仅训练这些新 Block）"""
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int, p_drop: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                           dropout=p_drop, activation="gelu", batch_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        # x: (B, T, D) -> (T, B, D)
        x_perm = x.transpose(0, 1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (B, T) True=pad
        y = self.encoder(x_perm, src_key_padding_mask=key_padding_mask)  # (T, B, D)
        y = y.transpose(0, 1)  # (B, T, D)
        return self.ln_out(y)


class TopLayerExpander(nn.Module):
    """若干 Residual MLP Layer 叠加在底座输出之上（极轻）"""
    def __init__(self, d_model: int, num_layers: int, mult: float = 2.0, p_drop: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([ResidualMLPLayer(d_model, mult=mult, p_drop=p_drop) for _ in range(num_layers)])

    def forward(self, x, attention_mask=None):
        for lyr in self.layers:
            x = lyr(x)
        return x


class ExpandedForMaskedLM(nn.Module):
    """
    底座 AutoModelForMaskedLM：
      roberta(**inputs) -> last_hidden_state -> [expansion] -> dropout -> lm_head -> logits
    仅训练 expansion（和可选的 lm_head）；底座（包括 encoder）全部冻结。
    """
    def __init__(self, base_mlm: AutoModelForMaskedLM, expansion: nn.Module | None,
                 tune_lm_head: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.base = base_mlm
        self.expansion = expansion
        self.tune_lm_head = bool(tune_lm_head)
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.config = self.base.config  # 暴露给 Trainer

        # 冻结底座 encoder
        if hasattr(self.base, "roberta"):
            for p in self.base.roberta.parameters():
                p.requires_grad = False
        else:
            for n, p in self.base.named_parameters():
                if not n.startswith("lm_head."):
                    p.requires_grad = False

        # lm_head 是否训练
        for p in self.base.lm_head.parameters():
            p.requires_grad = self.tune_lm_head

        # expansion 训练
        if self.expansion is not None:
            for p in self.expansion.parameters():
                p.requires_grad = True

        # Trainer 需要的标志
        self.is_gradient_checkpointing = False

    # ------- 供 Trainer 调用的接口（转发到底座） -------
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
    # ---------------------------------------------------

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
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


# ---------------------- 模式选择（含 Expansion） ----------------------
def setup_model_by_mode(base_model, args):
    """
    4 路：
      A) expansion!=none -> Block/Layer Expansion（优先级最高，忽略 LoRA/No-LoRA/Full）
      B) full_finetune
      C) no_lora（LN/bias/lm_head）
      D) LoRA（默认）
    """
    # A) Expansion（顶层叠加）
    if args.expansion != "none":
        print(f">>> Mode: EXPANSION = {args.expansion}")
        d_model = base_model.config.hidden_size
        nhead = getattr(base_model.config, "num_attention_heads", 8)
        dim_ff = int(d_model * args.expand_ffn_mult)

        if args.expansion == "block":
            expander = TopBlockExpander(
                d_model=d_model,
                nhead=args.expand_heads if args.expand_heads > 0 else nhead,
                num_layers=args.expand_layers,
                dim_ff=dim_ff,
                p_drop=args.expansion_dropout
            )
        elif args.expansion == "layer":
            expander = TopLayerExpander(
                d_model=d_model,
                num_layers=args.expand_layers,
                mult=args.expand_ffn_mult,
                p_drop=args.expansion_dropout
            )
        else:
            raise ValueError(f"Unknown expansion: {args.expansion}")

        model = ExpandedForMaskedLM(
            base_mlm=base_model,
            expansion=expander,
            tune_lm_head=bool(args.expansion_tune_lm_head),
            dropout_p=args.expansion_dropout
        )
        print_trainables(model)
        return model

    # B) Full finetune
    if args.full_finetune:
        for p in base_model.parameters():
            p.requires_grad = True
        print(">>> Mode: FULL finetune (no LoRA). All parameters trainable.")
        print_trainables(base_model)
        return base_model

    # C) 无 LoRA baseline
    if args.no_lora:
        freeze_all_params(base_model)
        optionally_unfreeze_layernorm_and_bias(base_model, enable_ln=args.unfreeze_ln, enable_bias=args.unfreeze_bias)
        if getattr(args, "unfreeze_lm_head", False):
            for n,p in base_model.named_parameters():
                if n.startswith("lm_head."):
                    p.requires_grad = True
        print(">>> Mode: NO LoRA baseline (freeze backbone; unfreeze LN/bias/lm_head as configured).")
        print_trainables(base_model)
        return base_model

    # D) LoRA（默认）
    freeze_all_params(base_model)
    optionally_unfreeze_layernorm_and_bias(base_model, enable_ln=args.unfreeze_ln, enable_bias=args.unfreeze_bias)
    lora_targets = tuple([t.strip() for t in args.lora_targets.split(",") if t.strip()])
    model = build_lora_model(base_model, target_modules=lora_targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.print_trainable_parameters()
    return model


# ---------------------- 稳健评测 ----------------------
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

    # 回退到手写评测
    loss = average_mlm_loss(
        trainer.model, eval_ds, data_collator,
        batch_size=trainer.args.per_device_eval_batch_size,
        num_workers=trainer.args.dataloader_num_workers
    )
    if loss is None:
        return {"loss": None, "ppl_approx": None}
    ppl = math.exp(loss) if loss < 20 else float("inf")
    return {"loss": float(loss), "ppl_approx": float(ppl)}


# ---------------------- 稳健保存（方法 2） ----------------------
def robust_save_model(trainer_model, tokenizer, out_dir, args, base_model_name_or_path):
    """
    - Expansion 模式：保存 expansion 的 state_dict 以及（可选）lm_head 的 state_dict（expansion.pt）
    - 其它模式（LoRA / full / no_lora）：调用 save_pretrained；若无该方法则保存 state_dict
    """
    save_dir = Path(out_dir) / (
        "adapter_or_expansion" if (args.expansion != "none" or (not args.no_lora and not args.full_finetune))
        else "model_final"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Expansion 检测：看是否有 base 与 expansion 属性
    is_expansion = (trainer_model.__class__.__name__ == "ExpandedForMaskedLM") or \
                   (hasattr(trainer_model, "base") and hasattr(trainer_model, "expansion"))

    if is_expansion:
        exp = getattr(trainer_model, "expansion", None)
        base = getattr(trainer_model, "base", None)
        tune_lm_head = bool(getattr(args, "expansion_tune_lm_head", False))

        payload = {
            "base_model_name_or_path": base_model_name_or_path,
            "expansion_type": args.expansion,         # "block" / "layer"
            "tune_lm_head": tune_lm_head,
            "train_args": {
                "expand_layers": args.expand_layers,
                "expand_ffn_mult": args.expand_ffn_mult,
                "expand_heads": args.expand_heads,
                "expansion_dropout": args.expansion_dropout,
            },
            "config": trainer_model.config.to_dict() if hasattr(trainer_model, "config") else {},
            "expansion_state_dict": (exp.state_dict() if exp is not None else None),
            "lm_head_state_dict": (base.lm_head.state_dict() if (tune_lm_head and base is not None) else None),
        }
        torch.save(payload, save_dir / "expansion.pt")
        if base is not None and hasattr(base, "config"):
            base.config.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(out_dir)
        print(f"[Saved Expansion] => {save_dir / 'expansion.pt'}")
        return

    # 其它模式
    if hasattr(trainer_model, "save_pretrained"):
        trainer_model.save_pretrained(str(save_dir))
    else:
        torch.save(trainer_model.state_dict(), save_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(out_dir)
    print(f"[Saved HF/Peft] => {save_dir}")


# ---------------------- 单次训练封装（支持指定是否 re-warm） ----------------------
def train_once(tokenizer, base_model_name_or_path, cpt_train, old_val, new_val, data_collator,
               args, out_dir, use_rewarm, before_old, before_new):
    """
    从 base_model_name_or_path 重新加载底座 -> 组装模式 -> (可选)GC -> 训练 -> 保存 -> 评测
    use_rewarm: False = 线性LR、无warmup；True = cosine + warmup_ratio
    """
    print(f"\n=== Train pass @ {out_dir} | re-warm = {use_rewarm} ===")
    base_model = AutoModelForMaskedLM.from_pretrained(base_model_name_or_path)

    # 组装（Expansion / LoRA / No-LoRA / Full）
    model = setup_model_by_mode(base_model, args)

    # GC 配置
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # 训练参数（方法 1：关闭 safetensors 中途保存）
    lr_scheduler_type = "cosine" if use_rewarm else "linear"
    warmup_ratio = args.warmup_ratio if use_rewarm else 0.0
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        save_safetensors=False,            # 关闭 safetensors，避免 tied weights 报错
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
        train_dataset=cpt_train,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    # 稳健保存（方法 2）
    robust_save_model(trainer.model, tokenizer, out_dir, args, base_model_name_or_path)

    # 训练后评测
    after_old = evaluate_loss(trainer, old_val, data_collator, desc="after_old")
    after_new = evaluate_loss(trainer, new_val, data_collator, desc="after_new")
    print(f"[After CPT @ {out_dir}] old_val: {after_old} | new_val: {after_new}")

    # 写本次 run 的 metrics_before_after.json
    def safe_diff(a, b): return (a - b) if (a is not None and b is not None) else None
    summary = {
        "config": {
            **vars(args),
            "model_name": base_model_name_or_path,
            "use_bf16": bool(args.bf16 and torch.cuda.is_available()),
            "use_rewarm": bool(use_rewarm),
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "output_dir": out_dir
        },
        "before": {"old_val": before_old, "new_val": before_new},
        "after":  {"old_val": after_old,  "new_val": after_new},
        "deltas": {
            "old_val_loss_diff": safe_diff(after_old["loss"], before_old["loss"]),
            "new_val_loss_diff": safe_diff(after_new["loss"], before_new["loss"])
        }
    }
    with open(Path(out_dir) / "metrics_before_after.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ---------------------- 主流程 ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--old_train", type=str, required=True)
    parser.add_argument("--old_val", type=str, required=True)
    parser.add_argument("--new_train", type=str, required=True)
    parser.add_argument("--new_val", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cpt_expansion")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)

    # re-warm (单跑) & 对比
    parser.add_argument("--rewarm_lr", action="store_true", help="单次训练使用 cosine+warmup（默认linear）")
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Expansion（新增）
    parser.add_argument("--expansion", type=str, default="none", choices=["none","block","layer"],
                        help="参数扩展：block=TransformerEncoder; layer=Residual MLP")
    parser.add_argument("--expand_layers", type=int, default=2, help="新加层数（Block/Layer 通用）")
    parser.add_argument("--expand_ffn_mult", type=float, default=2.0, help="Block 的 FFN 放大倍数 / Layer 的瓶颈倍数")
    parser.add_argument("--expand_heads", type=int, default=0, help="Block 的头数（0=跟随底座）")
    parser.add_argument("--expansion_dropout", type=float, default=0.1, help="Block/Layer 的 dropout")
    parser.add_argument("--expansion_tune_lm_head", action="store_true", help="扩展时同时训练 lm_head（推荐开启）")

    # LoRA / No-LoRA / Full
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_targets", type=str, default="query,value", help="如: query,value 或 query,key,value")
    parser.add_argument("--no_lora", action="store_true", help="禁用 LoRA，作为 baseline 对照")
    parser.add_argument("--full_finetune", action="store_true", help="全量微调（不冻结任何参数）")
    parser.add_argument("--unfreeze_ln", action="store_true", help="除 LoRA 外，放开 LayerNorm（轻量基线）")
    parser.add_argument("--unfreeze_bias", action="store_true", help="配合 --unfreeze_ln，放开 bias")
    parser.add_argument("--unfreeze_lm_head", action="store_true", help="no_lora 时放开 lm_head（更稳的新域适配）")

    # 回放
    parser.add_argument("--old_replay_ratio", type=float, default=0.0, help="训练时混入 old_train 的比例（0~0.5 合理）")

    # GC
    parser.add_argument("--gradient_checkpointing", action="store_true", help="开启梯度检查点（省显存）")
    parser.add_argument("--gc_use_reentrant", action="store_true", help="显式使用 reentrant 版本（默认关闭）")

    # 一键对比 re-warm & re-decay
    parser.add_argument("--compare_rewarm", action="store_true", help="一键对比不开 re-warm vs 开 re-warm")

    args = parser.parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(">>> Loading tokenizer & base model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    try:
        tokenizer.clean_up_tokenization_spaces = True
    except Exception:
        pass

    # 数据
    old_train_raw = load_text_dataset(args.old_train)
    old_val_raw   = load_text_dataset(args.old_val)
    new_train_raw = load_text_dataset(args.new_train)
    new_val_raw   = load_text_dataset(args.new_val)

    old_train = tokenize_dataset(old_train_raw, tokenizer, args.seq_len)
    old_val   = tokenize_dataset(old_val_raw,   tokenizer, args.seq_len)
    new_train = tokenize_dataset(new_train_raw, tokenizer, args.seq_len)
    new_val   = tokenize_dataset(new_val_raw,   tokenizer, args.seq_len)

    print(f"Dataset sizes -> old_train:{len(old_train)} old_val:{len(old_val)} new_train:{len(new_train)} new_val:{len(new_val)}")

    # 回放混合
    if args.old_replay_ratio > 0:
        n_new = len(new_train)
        n_old = int(n_new * args.old_replay_ratio)
        old_sample = old_train.shuffle(seed=args.seed).select(range(min(n_old, len(old_train))))
        cpt_train = concatenate_datasets([new_train, old_sample]).shuffle(seed=args.seed)
    else:
        cpt_train = new_train

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_prob
    )

    # 训练前“before”评测（用纯底座）
    base_model_for_before = AutoModelForMaskedLM.from_pretrained(args.model_name)
    training_args_probe = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=2,
        logging_strategy="no",
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16 and torch.cuda.is_available(),
        eval_strategy="no",
        save_safetensors=False  # 与主训练一致，禁用 safetensors
    )
    trainer_probe = Trainer(
        model=base_model_for_before, args=training_args_probe,
        data_collator=data_collator, tokenizer=tokenizer
    )
    before_old = evaluate_loss(trainer_probe, old_val, data_collator, desc="before_old")
    before_new = evaluate_loss(trainer_probe, new_val, data_collator, desc="before_new")
    print(f"[Before CPT] old_val: {before_old}  |  new_val: {before_new}")

    # 对比模式 vs 单跑
    if args.compare_rewarm:
        out_norewarm = str(Path(args.output_dir).with_suffix("")) + "_norewarm"
        out_rewarm   = str(Path(args.output_dir).with_suffix("")) + "_rewarm"

        # 1) baseline: 无 re-warm（linear，无 warmup）
        sum_no = train_once(
            tokenizer=tokenizer,
            base_model_name_or_path=args.model_name,
            cpt_train=cpt_train, old_val=old_val, new_val=new_val, data_collator=data_collator,
            args=args, out_dir=out_norewarm, use_rewarm=False,
            before_old=before_old, before_new=before_new
        )
        # 2) re-warm: cosine + warmup
        sum_rw = train_once(
            tokenizer=tokenizer,
            base_model_name_or_path=args.model_name,
            cpt_train=cpt_train, old_val=old_val, new_val=new_val, data_collator=data_collator,
            args=args, out_dir=out_rewarm, use_rewarm=True,
            before_old=before_old, before_new=before_new
        )

        # 写汇总对比
        compare = {
            "before": {"old_val": before_old, "new_val": before_new},
            "no_rewarm": sum_no,
            "rewarm": sum_rw,
            "improvements": {
                "old_val_delta_better": (sum_no["deltas"]["old_val_loss_diff"] - sum_rw["deltas"]["old_val_loss_diff"]) if (sum_no["deltas"]["old_val_loss_diff"] is not None and sum_rw["deltas"]["old_val_loss_diff"] is not None) else None,
                "new_val_delta_better": (sum_rw["deltas"]["new_val_loss_diff"] - sum_no["deltas"]["new_val_loss_diff"]) if (sum_no["deltas"]["new_val_loss_diff"] is not None and sum_rw["deltas"]["new_val_loss_diff"] is not None) else None
            }
        }
        with open(Path(args.output_dir) / "rewarm_comparison.json", "w", encoding="utf-8") as f:
            json.dump(compare, f, ensure_ascii=False, indent=2)
        print("\n=== Saved comparison to:", str(Path(args.output_dir) / "rewarm_comparison.json"))
    else:
        # 单跑：遵循 --rewarm_lr 的设置
        out_dir = args.output_dir
        _ = train_once(
            tokenizer=tokenizer,
            base_model_name_or_path=args.model_name,
            cpt_train=cpt_train, old_val=old_val, new_val=new_val, data_collator=data_collator,
            args=args, out_dir=out_dir, use_rewarm=bool(args.rewarm_lr),
            before_old=before_old, before_new=before_new
        )


if __name__ == "__main__":
    main()
