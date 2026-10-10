"""Mission 03：加载一个小模型，用 model.generate() 生成文本，并测速。

覆盖验收项：输入token数 / 输出token数 / 生成时间 / tokens/s。
这是"最简单能跑"的版本；Mission 04 用的手写循环在 src/minillm/engine.py。

用法：
    python scripts/run_generation.py
    python scripts/run_generation.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "用一句话介绍你自己" --max-new-tokens 32

候选模型（本地能跑）：
    Qwen/Qwen2.5-0.5B-Instruct          # 推荐，CPU/GPU 都流畅
    Qwen/Qwen2.5-1.5B-Instruct          # 质量更好，RTX 5060 无压力
    HuggingFaceTB/SmolLM2-135M-Instruct # 更小更快，纯学习用
    meta-llama/Llama-3.2-1B-Instruct    # 需先在 HF 上同意许可并配置 token
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Mission 03：model.generate() 加载模型并测速")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="用一句话介绍你自己")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto")  # auto / cuda / cpu
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[1/4] 加载模型: {args.model} (device={device}) ...")

    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    load_s = time.perf_counter() - t_load
    print(f"[2/4] 模型加载完成，耗时 {load_s:.1f}s")

    # chat 格式：Qwen/Llama/SmolLM 都有 chat_template
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(enc["input_ids"].shape[1])
    print(f"[3/4] prompt={args.prompt!r}  →  输入 token 数 = {prompt_tokens}")

    t_gen = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_s = time.perf_counter() - t_gen

    new_ids = out[0][prompt_tokens:]
    completion_tokens = int(new_ids.shape[0])
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    tokens_per_s = completion_tokens / gen_s if gen_s > 0 else 0.0

    print("[4/4] 生成结果:")
    print(f"  {text}")
    print("=" * 56)
    print(f"  输入 token 数 : {prompt_tokens}")
    print(f"  输出 token 数 : {completion_tokens}")
    print(f"  生成耗时      : {gen_s * 1000:.0f} ms")
    print(f"  吞吐          : {tokens_per_s:.1f} tokens/s")
    print(f"  每 token 平均 : {gen_s / completion_tokens * 1000:.1f} ms（≈ TPOT，Mission 05 再正式测）")


if __name__ == "__main__":
    main()
