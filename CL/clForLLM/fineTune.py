# clForLLM.py
# Cross-platform (Windows/macOS/Linux) Instruction Tuning demo with before/after comparison.
# - Default model: EleutherAI/pythia-70m-deduped (has .safetensors)
# - Uses DataCollatorForLanguageModeling for padding & labels
# - Auto device: CUDA -> MPS -> CPU

import os
# 跳过 torchvision（纯文本任务），并允许回退（在 macOS MPS 上更稳；Windows 也不受影响）
os.environ["TRANSFORMERS_NO_TF"] = "1"           # 禁用 TensorFlow 相关导入
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # 禁用 torchvision 可选导入（我们是纯文本）
os.environ["TRANSFORMERS_NO_FLAX"] = "1"         # 禁用 Flax/JAX
os.environ["PYTHONNOUSERSITE"] = "1"             # 不加载用户目录的 site-packages，隔离全局 pip 包
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from typing import Dict, Any
import random

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if name == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # auto: 优先 CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_text_from_dolly(ex: Dict[str, Any]) -> str:
    """把 Dolly 的 (instruction/context/response) 合成一段简单的 IT 文本。"""
    instr = (ex.get("instruction") or "").strip()
    ctx   = (ex.get("context") or "").strip()
    resp  = (ex.get("response") or "").strip()
    parts = [instr]
    if ctx:
        parts.append(ctx)
    parts.append(resp)
    return "\n".join(parts)

@torch.no_grad()
def generate_once(model, tok, prompt: str, device, max_new_tokens=160, temperature=0.8, top_k=50):
    model.eval()
    model.to(device)
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="EleutherAI/pythia-410m-deduped",
                    help="Safetensors model, e.g. EleutherAI/pythia-410m-deduped")
    ap.add_argument("--dataset", type=str, default="databricks/databricks-dolly-15k")
    ap.add_argument("--train_size", type=int, default=200, help="subset size for a quick demo")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out_dir", type=str, default="./it_demo_out")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=42)
    # generation / comparison
    ap.add_argument("--prompt", type=str, default="用三句话解释粒子滤波和卡尔曼滤波的主要区别。")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[info] device = {device}")

    # --- Tokenizer & Model (safetensors-first) ---
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,      # 避免 torch.load 加载 .bin
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # --- BEFORE: 基线生成（微调前） ---
    print("\n========== BEFORE FINE-TUNE ==========")
    baseline_text = generate_once(
        model, tok, args.prompt, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(baseline_text)

    # --- 加载数据（抽小集，Windows 上也跑得动） ---
    ds = load_dataset(args.dataset)["train"]
    if args.train_size and args.train_size < len(ds):
        ds = ds.select(range(args.train_size))

    # 预处理：只做截断；padding/labels 交给 collator
    def preprocess(ex):
        text = build_text_from_dolly(ex)
        return tok(text, truncation=True, max_length=args.max_len)

    ds_proc = ds.map(preprocess, remove_columns=ds.column_names)

    # Collator：统一 padding，并构建 labels（padding 位置为 -100）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        bf16=False, fp16=False,          # Windows 上大多 CPU 或 CUDA；小模型 fp32 已足够
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc,
        tokenizer=tok,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[ok] model saved to: {args.out_dir}")

    # --- AFTER: 微调后生成（同一 prompt，对比效果） ---
    print("\n========== AFTER  FINE-TUNE ==========")
    after_text = generate_once(
        model, tok, args.prompt, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(after_text)

if __name__ == "__main__":
    main()
