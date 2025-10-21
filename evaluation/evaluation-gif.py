#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch evaluate GIF similarity (first & last frame) from TWO folders via LLM.

用法示例：
python gif_frame_eval.py \
  --api-base "https://api-gateway.glm.ai/v1" \
  --api-key "$API_KEY" \
  --model-name "gpt-5-2025-08-07" \
  --gt-dir /path/to/gt_gifs \
  --gen-dir /path/to/gen_gifs \
  --need-dir ./eval_work \
  --json-output-dir ./eval_out \
  --repeat-try --max-repeat-times 1 \
  --temperature 0 --top-p 0.95 --max-completion-tokens 4096
"""

import os
import re
import json
import math
import time
import base64
import argparse
from pathlib import Path
from typing import List, Dict
from multiprocessing import Process
from PIL import Image
import requests

IMG_EXTS = {".gif"}
TEMP_FRAME_DIR = "temp_frames"

# ----------------------------
# Prompt for LLM
# ----------------------------
DEFAULT_PROMPT = (
    "You have two animated GIFs, each represented by their first and last frames (four images in total): "
    "Reference GIF first frame, Reference GIF last frame, Candidate GIF first frame, Candidate GIF last frame. "
    "Please determine whether the Candidate GIF is visually similar to the Reference GIF based on these frames. "
    "Provide a brief explanation of your reasoning. "
    "You must rate the response on a scale of 1 to 100 by strictly following this format:\n"
    "\"Rating: [[85]]\".\n"
    "Do not output the score in any other form."
)

# ----------------------------
# Utilities
# ----------------------------
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def safe_mkdirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def extract_first_last_frame(gif_path: str, tmp_dir: str) -> List[str]:
    """
    提取 GIF 的首尾帧并保存为 PNG
    返回两帧路径列表
    """
    safe_mkdirs(tmp_dir)
    try:
        with Image.open(gif_path) as im:
            frames = []
            frames.append(im.convert("RGBA"))
            last_frame_index = im.n_frames - 1
            if last_frame_index > 0:
                im.seek(last_frame_index)
                frames.append(im.convert("RGBA"))
            else:
                frames.append(im.convert("RGBA"))
            frame_paths = []
            for idx, frame in enumerate(frames):
                frame_file = os.path.join(tmp_dir, f"{Path(gif_path).stem}_frame{idx}.png")
                frame.save(frame_file)
                frame_paths.append(frame_file)
            return frame_paths
    except Exception as e:
        print(f"GIF 提取失败: {gif_path}, error={e}")
        return []

def guess_mime_type(p: str) -> str:
    return "image/png"

def do_one_request(payload: dict, api_base: str, api_key: str, timeout: int = 1200) -> Dict:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), "ok"
        else:
            return None, f"fail:{resp.status_code}-{resp.text}"
    except Exception as e:
        return None, f"fail:{e}"

def make_dual_image_content(prompt_text: str, ref_paths: List[str], cand_paths: List[str]) -> list:
    """
    构建多模态输入内容：首尾两帧
    """
    content = [{"type": "text", "text": prompt_text}]
    content.append({"type": "text", "text": "Reference GIF frames:"})
    for ref in ref_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{guess_mime_type(ref)};base64,{encode_image_to_base64(ref)}"}
        })
    content.append({"type": "text", "text": "Candidate GIF frames:"})
    for cand in cand_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{guess_mime_type(cand)};base64,{encode_image_to_base64(cand)}"}
        })
    return content

def build_payload(model: str, content: list, temperature: float, top_p: float, max_tokens: int, use_openai_max_completion_tokens: bool) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": temperature, "top_p": top_p}
    if use_openai_max_completion_tokens:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    return payload

def normalize_key(p: Path) -> str:
    s = p.stem.lower()
    s = s.replace(" ", "")
    return s

def scan_folder(root: Path) -> Dict[str, str]:
    mp = {}
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
            key = normalize_key(fp)
            if key not in mp or len(str(fp)) < len(mp[key]):
                mp[key] = str(fp.resolve())
    return mp

def build_pairs_from_dirs(gt_dir: str, gen_dir: str) -> (List[dict], List[str], List[str]):
    gt_map = scan_folder(Path(gt_dir))
    gen_map = scan_folder(Path(gen_dir))
    keys_inter = sorted(set(gt_map.keys()) & set(gen_map.keys()))
    inputs = [{"pair_id": k, "reference": gt_map[k], "candidate": gen_map[k]} for k in keys_inter]
    miss_gt = sorted(set(gen_map.keys()) - set(gt_map.keys()))
    miss_gen = sorted(set(gt_map.keys()) - set(gen_map.keys()))
    return inputs, miss_gt, miss_gen

# ----------------------------
# Core processing
# ----------------------------
def process_one(input_row: dict, worker_id: int, args: argparse.Namespace, output_file: str, index_file: str, error_idxs: List[int], idx: int, prompt_text: str):
    pair_id = input_row.get("pair_id", str(idx))
    ref_gif = input_row["reference"]
    cand_gif = input_row["candidate"]

    # 提取首尾帧
    ref_frames = extract_first_last_frame(ref_gif, TEMP_FRAME_DIR)
    cand_frames = extract_first_last_frame(cand_gif, TEMP_FRAME_DIR)
    if not ref_frames or not cand_frames:
        print(f"[Worker {worker_id}] GIF 提取失败 idx={idx}")
        if error_idxs is not None:
            error_idxs.append(idx)
        return

    content = make_dual_image_content(prompt_text, ref_frames, cand_frames)
    payload = build_payload(model=args.model_name, content=content, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_completion_tokens, use_openai_max_completion_tokens=args.use_openai_max_completion_tokens)
    print(f"[Worker {worker_id}] pair_id={pair_id} 请求开始")
    response, status = do_one_request(payload, api_base=args.api_base, api_key=args.api_key, timeout=args.timeout)

    if status == "ok" and response:
        message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        result = dict(input_row)
        result.update({"llm_raw_output": message_content.strip()})
        with open(output_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
        with open(index_file, 'w') as fidx:
            fidx.write(str(idx + 1))
        print(f"[Worker {worker_id}] Done idx={idx}")
    else:
        print(f"[Worker {worker_id}] 请求失败 idx={idx}, reason={status}")
        if error_idxs is not None:
            error_idxs.append(idx)

def worker_fn(worker_id: int, inputs: List[dict], start_idx: int, end_idx: int, args: argparse.Namespace, index_prefix: str, output_prefix: str, prompt_text: str):
    index_file = f"{index_prefix}_{worker_id}.txt"
    output_file = f"{output_prefix}_{worker_id}.jsonl"
    if os.path.exists(index_file):
        try:
            persisted_idx = open(index_file, 'r').read().strip()
            if persisted_idx.isdigit():
                start_idx = int(persisted_idx)
        except Exception:
            pass
    error_idxs = []
    for idx in range(start_idx, end_idx):
        process_one(inputs[idx], worker_id, args, output_file, index_file, error_idxs, idx, prompt_text)
    if args.repeat_try:
        repeat_times = 0
        while repeat_times < args.max_repeat_times and error_idxs:
            print(f"[Worker {worker_id}] 重试第 {repeat_times+1} 轮，失败样本数={len(error_idxs)}")
            new_error_idxs = []
            for idx in error_idxs:
                process_one(inputs[idx], worker_id, args, output_file, index_file, new_error_idxs, idx, prompt_text)
            error_idxs = new_error_idxs
            repeat_times += 1

def run_multiprocess(inputs: List[dict], args: argparse.Namespace, need_dir_round: str, json_out_round: str, prompt_text: str):
    total = len(inputs)
    if total == 0:
        print("无输入数据，跳过本轮。")
        return
    chunk_size = math.ceil(total / args.num_workers)
    processes = []
    for i in range(args.num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        p = Process(target=worker_fn, args=(i, inputs, start, end, args,
                                            os.path.join(need_dir_round, "data_index_worker"),
                                            os.path.join(json_out_round, "eval_worker"), prompt_text))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def merge_jsonl_files(file_list: List[str], output_file: str):
    merged_count = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for file in file_list:
            if not os.path.exists(file):
                continue
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        merged_count += 1
    print(f"总合并完成！共写入 {merged_count} 条记录 -> {output_file}")

# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GIF similarity (first & last frame) via LLM")
    parser.add_argument("--api-base", type=str, default="https://api-gateway.glm.ai/v1")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model-name", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--use_openai_max_completion_tokens", action="store_true")
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--need-dir", type=str, required=True)
    parser.add_argument("--json-output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--repeat-try", action="store_true")
    parser.add_argument("--max-repeat-times", type=int, default=1)
    parser.add_argument("--loop-max-times", type=int, default=1)
    parser.add_argument("--continue-from-loop", type=int, default=1)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    return parser.parse_args()

def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file and os.path.exists(args.prompt_file):
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT

def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("缺少 API Key。")

    prompt_text = load_prompt(args)
    safe_mkdirs(args.need_dir, args.json_output_dir, TEMP_FRAME_DIR)

    inputs, miss_gt, miss_gen = build_pairs_from_dirs(args.gt_dir, args.gen_dir)
    if args.test_limit > 0:
        inputs = inputs[:args.test_limit]

    TOTAL = len(inputs)
    print(f"共配对到 {TOTAL} 对 GIF。未在GT中的键数：{len(miss_gt)}；未在GEN中的键数：{len(miss_gen)}")
    if TOTAL == 0:
        Path(os.path.join(args.need_dir, "missing_in_gt.txt")).write_text("\n".join(miss_gt), encoding="utf-8")
        Path(os.path.join(args.need_dir, "missing_in_gen.txt")).write_text("\n".join(miss_gen), encoding="utf-8")
        return

    loop_times = args.continue_from_loop if args.continue_from_loop > 1 else 1
    cur_inputs = inputs
    original_total = TOTAL

    while loop_times <= args.loop_max_times:
        print(f"第 {loop_times} 次循环，模型：{args.model_name}")
        need_dir_round = os.path.join(args.need_dir, str(loop_times))
        json_out_round = os.path.join(args.json_output_dir, str(loop_times))
        safe_mkdirs(need_dir_round, json_out_round)

        cur_total = len(cur_inputs)
        if cur_total == 0:
            print("本轮没有可处理数据，结束。")
            break

        remain_ratio = cur_total / original_total
        print(f"当前剩余数据量 {cur_total}，剩余比例 {remain_ratio:.4f}")

        run_multiprocess(cur_inputs, args, need_dir_round, json_out_round, prompt_text)

        # 合并输出
        final_list = [os.path.join(json_out_round, f) for f in os.listdir(json_out_round) if f.endswith(".jsonl")]
        merge_jsonl_files(final_list, os.path.join(json_out_round, "merged.jsonl"))

        # 形成 remained
        done_keys = set()
        with open(os.path.join(json_out_round, "merged.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done_keys.add(obj.get("pair_id"))
        remained = [r for r in cur_inputs if r.get("pair_id") not in done_keys]
        Path(os.path.join(json_out_round, "remained.jsonl")).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in remained), encoding="utf-8"
        )

        cur_inputs = remained
        loop_times += 1
        if args.test_limit > 0:
            break

    # 汇总所有轮次
    final_jsons = [os.path.join(args.json_output_dir, str(i), "merged.jsonl") for i in range(1, loop_times)]
    merge_jsonl_files(final_jsons, os.path.join(args.json_output_dir, "final_merged.jsonl"))

    # 输出未配对列表
    Path(os.path.join(args.json_output_dir, "missing_in_gt.txt")).write_text("\n".join(miss_gt), encoding="utf-8")
    Path(os.path.join(args.json_output_dir, "missing_in_gen.txt")).write_text("\n".join(miss_gen), encoding="utf-8")
    print("处理完成！")

if __name__ == "__main__":
    main()
