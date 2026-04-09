#!/usr/bin/env python3
"""MLX inference bridge — loads model + optional LoRA adapter, generates text.

Protocol: reads JSON request from stdin, writes JSON response to stdout.

Request:
    {
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "adapter_path": "/path/to/adapters",  // optional
        "prompt": "user prompt text",
        "system_prompt": "system prompt text",  // optional
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.92,
        "repetition_penalty": 1.05
    }

Response:
    {
        "text": "generated text",
        "prompt_tokens": 123,
        "generation_tokens": 456,
        "generation_tps": 12.3,
        "peak_memory_gb": 18.5,
        "finish_reason": "stop"
    }
"""
import json
import sys
import time


def main():
    req = json.loads(sys.stdin.read())

    model_path = req["model"]
    adapter_path = req.get("adapter_path")
    prompt_text = req["prompt"]
    system_prompt = req.get("system_prompt")
    max_tokens = req.get("max_tokens", 2048)
    temperature = req.get("temperature", 0.7)
    top_p = req.get("top_p", 0.92)
    repetition_penalty = req.get("repetition_penalty", 1.05)

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Build chat messages for chat-template models (Gemma 4, etc.)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        # Fallback: concat system + user
        formatted = ""
        if system_prompt:
            formatted += system_prompt + "\n\n"
        formatted += prompt_text

    sampler = make_sampler(temp=temperature, top_p=top_p)

    t0 = time.time()

    full_text = ""
    last_resp = None
    for resp in stream_generate(
        model,
        tokenizer,
        formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        full_text += resp.text
        last_resp = resp

    elapsed = time.time() - t0

    result = {
        "text": full_text.strip(),
        "prompt_tokens": last_resp.prompt_tokens if last_resp else 0,
        "generation_tokens": last_resp.generation_tokens if last_resp else 0,
        "generation_tps": last_resp.generation_tps if last_resp else 0.0,
        "peak_memory_gb": last_resp.peak_memory if last_resp else 0.0,
        "finish_reason": last_resp.finish_reason if last_resp else "unknown",
        "elapsed_ms": int(elapsed * 1000),
    }

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
