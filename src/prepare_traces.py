"""
Prepares calibration and evaluation traces for the experiments in `experements/`:
1. Converts the independent evaluation traces `nemotron_trace.json` (48 dialogues: SWE, Agentic, Math, Chat)
   and `multilingual_trace.json` (52 dialogues: 16 Russian + 9 other languages) from Qwen tokens to target-model tokens
   (Llama-3.2), preserving the exact boundaries of the scored assistant responses (`score_spans`) and excluding
   the templated `<think>...</think>` prefixes.
2. Builds a balanced calibration trace from `calib_splits2/chat/data.jsonl` and `calib_splits2/agents/data.jsonl`
   (128 rows of 2048 tokens = 262 144 tokens):
   - saves `cal_trace.safetensors` (compatible with `exllamav3 --cal_data` and our `LDLQ-LIQK`),
   - saves `calib_imatrix.txt` for the stock `llama-imatrix`.
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


QWEN_TOK_DIR = os.path.expanduser("~/lm-inference/models/Qwen3.8-Flash-Next/SC2-EXL3-4.05")
SC_EVAL_DIR = os.path.expanduser("~/lm-inference/sc-quant/eval")
SC_CALIB_DIR = os.path.expanduser("~/lm-inference/sc-quant/hf/calib_splits2")


# Chat markup of the target model: llama3 (default), qwen (native, `--keep_qwen_markers`) or gemma4
CHAT_FORMAT = "llama3"
# <bos> token that must be explicitly prepended to every text (the Gemma 4 tokenizer does not add it itself)
FORCE_BOS_ID = None


def clean_qwen_chat_markers(text: str) -> str:
    """
    Replaces Qwen template special markers (`<|im_start|>role\\n`, `<|im_end|>`)
    with the Llama-3 format (`<|start_header_id|>role<|end_header_id|>\\n\\n`, `<|eot_id|>`)
    so that the model sees its native dialogue markup.
    """
    if CHAT_FORMAT == "qwen":
        return text
    if CHAT_FORMAT == "gemma4":
        text = re.sub(r"<\|im_start\|>assistant\n", "<|turn>model\n", text)
        text = re.sub(r"<\|im_start\|>(system|user|tool)\n", r"<|turn>\1\n", text)
        return text.replace("<|im_end|>", "<turn|>")
    text = re.sub(
        r"<\|im_start\|>(system|user|assistant|tool)\n",
        r"<|start_header_id|>\1<|end_header_id|>\n\n",
        text,
    )
    text = text.replace("<|im_end|>", "<|eot_id|>")
    return text


def strip_forced_think_prefix(span_text: str) -> Tuple[str, str]:
    """
    If the scored assistant response starts with a templated empty block `<think>\\n\\n</think>\\n\\n`
    or an opening `<think>\\n` tag, returns `(prefix_to_move_to_prompt, clean_scored_text)`.
    """
    m = re.match(r"^(\s*<think>\s*</think>\s*|\s*<think>\n)", span_text)
    if m:
        cut = m.end()
        return span_text[:cut], span_text[cut:]
    return "", span_text


def convert_eval_trace(
    src_path: str,
    dst_path: str,
    qwen_tok,
    target_tok,
    max_seq_len: int = 8192,
) -> Dict:
    raw = json.load(open(src_path, "r", encoding="utf-8"))
    out_rows = []
    total_scored = 0

    for row in raw["rows"]:
        seq = row["input_ids"] + row["response_ids"]
        spans = row["score_spans"]
        if not spans:
            continue

        # Split the sequence into alternating unscored and scored segments,
        # decode each segment with the Qwen tokenizer, strip the templated <think> and encode with the target tokenizer
        cursor = 0
        target_ids: List[int] = []
        target_spans: List[List[int]] = []

        for a, b in spans:
            if a > cursor:
                unscored_txt = qwen_tok.decode(seq[cursor:a], skip_special_tokens=False)
                unscored_txt = clean_qwen_chat_markers(unscored_txt)
                target_ids.extend(target_tok.encode(unscored_txt, add_special_tokens=(cursor == 0)))
                if cursor == 0 and FORCE_BOS_ID is not None and target_ids[:1] != [FORCE_BOS_ID]:
                    target_ids.insert(0, FORCE_BOS_ID)

            scored_txt = qwen_tok.decode(seq[a:b], skip_special_tokens=False)
            think_prefix, clean_scored = strip_forced_think_prefix(scored_txt)
            if think_prefix:
                target_ids.extend(target_tok.encode(clean_qwen_chat_markers(think_prefix), add_special_tokens=False))

            clean_scored = clean_qwen_chat_markers(clean_scored)
            scored_seg_ids = target_tok.encode(clean_scored, add_special_tokens=False)
            if scored_seg_ids:
                s_start = len(target_ids)
                target_ids.extend(scored_seg_ids)
                s_end = len(target_ids)
                target_spans.append([s_start, s_end])
            cursor = b

        if cursor < len(seq):
            tail_txt = clean_qwen_chat_markers(qwen_tok.decode(seq[cursor:], skip_special_tokens=False))
            target_ids.extend(target_tok.encode(tail_txt, add_special_tokens=False))

        # If the row is longer than max_seq_len, carefully truncate it from the end or skip empty spans
        if len(target_ids) > max_seq_len:
            target_ids = target_ids[:max_seq_len]
            clipped_spans = []
            for s0, s1 in target_spans:
                if s0 < max_seq_len:
                    s1_c = min(s1, max_seq_len)
                    if s1_c - s0 >= 16:
                        clipped_spans.append([s0, s1_c])
            target_spans = clipped_spans

        if not target_spans:
            continue

        first_span_start = target_spans[0][0]
        total_scored += sum(s1 - s0 for s0, s1 in target_spans)
        out_rows.append(
            {
                "conversation": row.get("conversation", 0),
                "turn": row.get("turn", 0),
                "source": row.get("source", "default"),
                "uuid": row.get("uuid", ""),
                "input_ids": target_ids[:first_span_start],
                "response_ids": target_ids[first_span_start:],
                "score_spans": target_spans,
            }
        )

    out_data = {
        "model": target_tok.name_or_path,
        "vocab_size": len(target_tok),
        "meta": {"source_trace": src_path, "rows": len(out_rows), "scored_tokens": total_scored},
        "rows": out_rows,
    }
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"Saved trace {dst_path}: {len(out_rows)} dialogues, {total_scored} scored tokens")
    return out_data


def build_calibration_data(
    target_tok,
    out_safetensors: str,
    out_txt: str,
    n_rows: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
):
    rng = random.Random(seed)
    chat_path = os.path.join(SC_CALIB_DIR, "chat", "data.jsonl")
    agent_path = os.path.join(SC_CALIB_DIR, "agents", "data.jsonl")

    texts: List[str] = []
    for path, weight in [(chat_path, 0.65), (agent_path, 0.35)]:
        file_texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                txt = rec.get("text", "")
                if txt and len(txt) >= 500:
                    file_texts.append(clean_qwen_chat_markers(txt))
        rng.shuffle(file_texts)
        take = max(1, int(n_rows * 2 * weight))
        texts.extend(file_texts[:take])

    rng.shuffle(texts)

    # Build a contiguous token stream and cut it into n_rows blocks of seq_len tokens
    token_stream: List[int] = []
    txt_chunks: List[str] = []
    needed_tokens = n_rows * seq_len

    for txt in texts:
        ids = target_tok.encode(txt, add_special_tokens=True)
        if FORCE_BOS_ID is not None and ids[:1] != [FORCE_BOS_ID]:
            ids = [FORCE_BOS_ID] + ids
        token_stream.extend(ids)
        txt_chunks.append(txt)
        if len(token_stream) >= needed_tokens:
            break

    assert len(token_stream) >= needed_tokens, f"Not enough tokens: {len(token_stream)} < {needed_tokens}"
    token_tensor = torch.tensor(token_stream[:needed_tokens], dtype=torch.int64).reshape(n_rows, seq_len)
    save_file({"input_ids": token_tensor}, out_safetensors)

    # For llama-imatrix, save the first ~64 texts (about 100-150k tokens)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(txt_chunks[:64]))

    print(
        f"Saved calibration: {out_safetensors} ({n_rows}x{seq_len} = {needed_tokens} tokens) and {out_txt}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cal_rows", type=int, default=128)
    ap.add_argument("--cal_len", type=int, default=2048)
    ap.add_argument("--keep_qwen_markers", action="store_true", help="do not convert Qwen markup to Llama-3")
    ap.add_argument("--chat_format", choices=["llama3", "qwen", "gemma4"], default=None)
    args = ap.parse_args()
    global CHAT_FORMAT, FORCE_BOS_ID
    CHAT_FORMAT = args.chat_format or ("qwen" if args.keep_qwen_markers else "llama3")

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading Qwen and target-model tokenizers...")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_TOK_DIR)
    target_tok = AutoTokenizer.from_pretrained(args.model_dir)
    if CHAT_FORMAT == "gemma4":
        FORCE_BOS_ID = target_tok.bos_token_id

    convert_eval_trace(
        os.path.join(SC_EVAL_DIR, "nemotron_trace.json"),
        os.path.join(args.out_dir, "nemotron_trace.json"),
        qwen_tok,
        target_tok,
    )
    convert_eval_trace(
        os.path.join(SC_EVAL_DIR, "multilingual_trace.json"),
        os.path.join(args.out_dir, "multilingual_trace.json"),
        qwen_tok,
        target_tok,
    )
    build_calibration_data(
        target_tok,
        os.path.join(args.out_dir, "cal_trace.safetensors"),
        os.path.join(args.out_dir, "calib_imatrix.txt"),
        n_rows=args.cal_rows,
        seq_len=args.cal_len,
    )


if __name__ == "__main__":
    main()

