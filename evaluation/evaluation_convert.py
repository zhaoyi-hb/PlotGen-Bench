#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch evaluate image similarity from TWO folders (GT vs GEN) via LLM.

支持子目录层级匹配，例如：
gt-dir/folderA/code_B.png 与 gen-dir/folderB/code_B.png 会被分别识别。
"""

import os
import re
import json
import math
import time
import base64
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from multiprocessing import Process
import requests

# ----------------------------
# Default scoring prompt
# ----------------------------
DEFAULT_PROMPT = (
    """
    You are an expert in verifying chart data consistency. Your task is to strictly compare two images (Original vs Rendered (AI generated)) and, based on the following evaluation criteria, provide a score for each item, note any differences (if applicable), and finally give the total score (out of 100) along with the overall grade.
Evaluation Criteria:
1.Chart Type Accuracy (10 points): Determine whether the overall type of the rendered chart is identical to the original.
2.Element Completeness (20 points): Check if all elements from the original (data points, axes, legends, labels, etc.) are fully preserved in the rendered chart, with no omissions or additions.
3.Element Accuracy (15 points): Verify whether the shape, number, and structure of each element are identical to the original.
4.Position Alignment (10 points): Confirm whether the positions and relative layouts of all elements are consistent with the original.
5.Size and Proportion (10 points): Ensure that the overall chart size, axis scales, and boundary ranges match the original.
6.Color and Style Consistency (20 points): Check whether fill colors, line colors, background colors, line thickness, transparency, gradients, etc. are consistent with the original.
7.Text Consistency (10 points): Verify whether titles, axis labels, legend text, font sizes, font types, and alignments are consistent with the original.
8.Image Dimensions (5 points): Confirm whether the resolution, width, and height of the image match the original.
###Evaluation
After providing your explanation, you must rate the response on a scale of 1 to 100 by strictly following this format:
"Rating: [[85]]". Do not output the score in any other form.
    """
)

RATING_RE = re.compile(r"Rating:\s*\[\[(\d{1,3})\]\]", re.IGNORECASE)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# ----------------------------
# Utilities
# ----------------------------
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def guess_mime_type(p: str) -> str:
    ext = Path(p).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext in [".webp"]:
        return "image/webp"
    if ext in [".bmp"]:
        return "image/bmp"
    if ext in [".gif"]:
        return "image/gif"
    return "image/png"

def do_one_request(payload: dict, api_base: str, api_key: str, timeout: int = 1200) -> Tuple[Optional[dict], str]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    start_time = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.time() - start_time
        if resp.status_code == 200:
            print(f"请求成功，耗时{elapsed:.2f}秒")
            return resp.json(), "ok"
        else:
            msg = f"code {resp.status_code} - {resp.text}"
            print(f"请求失败: {msg}")
            return None, f"fail:{msg}"
    except Exception as e:
        print(f"请求异常: {e}")
        return None, f"fail:{e}"

def safe_mkdirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file and os.path.exists(args.prompt_file):
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT

def normalize_key(p: Path, root: Path) -> str:
    """
    使用相对路径（去掉后缀）作为 key，保留子目录结构
    例如 gt-dir/folderA/code_B.png -> folderA/code_b
    """
    rel = p.relative_to(root).with_suffix("")  # 去掉后缀
    s = str(rel).replace("\\", "/").lower()    # 用 / 统一分隔符并小写
    return s
def scan_folder(root: Path) -> Dict[str, str]:
    """
    扫描 root 下所有子目录，找到文件名为 code_B.png 的文件。
    key 使用相对路径（去掉后缀）保留子目录层级。
    """
    mp: Dict[str, str] = {}
    for subdir in root.iterdir():
        if subdir.is_dir():
            for fp in subdir.rglob("code_B.png"):  # 只匹配 code_B.png
                key = normalize_key(fp, root)
                if key not in mp:
                    mp[key] = str(fp.resolve())
                else:
                    # 简单冲突处理：选择路径更短的
                    if len(str(fp)) < len(mp[key]):
                        mp[key] = str(fp.resolve())
    return mp

def build_pairs_from_dirs(gt_dir: str, gen_dir: str) -> Tuple[List[dict], List[str], List[str]]:
    gt_map = scan_folder(Path(gt_dir))
    gen_map = scan_folder(Path(gen_dir))
    
    keys_inter = sorted(set(gt_map.keys()) & set(gen_map.keys()))
    inputs = [{"pair_id": k, "reference": gt_map[k], "candidate": gen_map[k]} for k in keys_inter]

    miss_gt = sorted(set(gen_map.keys()) - set(gt_map.keys()))
    miss_gen = sorted(set(gt_map.keys()) - set(gen_map.keys()))
    return inputs, miss_gt, miss_gen

def parse_llm_output(text: str) -> Tuple[str, Optional[int]]:
    if text is None:
        return "", None
    rating = None
    m = RATING_RE.search(text)
    if m:
        try:
            rating = int(m.group(1))
        except Exception:
            rating = None
    explanation = RATING_RE.sub("", text).strip()
    return explanation, rating

# ----------------------------
# Build payload
# ----------------------------
def build_payload(
    model: str,
    content: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_openai_max_completion_tokens: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": top_p,
    }
    if use_openai_max_completion_tokens:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    return payload

def make_dual_image_content(prompt_text: str, ref_path: str, cand_path: str) -> list:
    
    ref_path_source = ref_path.replace("code_B.png","code_A.png")
    
    ref_mime = guess_mime_type(ref_path)
    cand_mime = guess_mime_type(cand_path)
    
    ref_b64_source = encode_image_to_base64(ref_path_source)
    ref_b64 = encode_image_to_base64(ref_path)
    cand_b64 = encode_image_to_base64(cand_path)
    return [
        {"type": "text", "text": prompt_text},
        # {"type": "text", "text": "Reference source image:"},
        # {"type": "image_url", "image_url": {"url": f"data:{ref_mime};base64,{ref_b64_source}" }},
        {"type": "text", "text": "Reference image (ground truth):"},
        {"type": "image_url", "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}" }},
        {"type": "text", "text": "Candidate image (AI generated):"},
        {"type": "image_url", "image_url": {"url": f"data:{cand_mime};base64,{cand_b64}" }},
    ]

# ----------------------------
# Core per-item request
# ----------------------------
def process_one(
    input_row: dict,
    worker_id: int,
    args: argparse.Namespace,
    output_file: str,
    index_file: str,
    error_idxs: List[int],
    idx: int,
    prompt_text: str,
):
    pair_id = input_row.get("pair_id", str(idx))
    ref_path = input_row["reference"]
    cand_path = input_row["candidate"]

    if not (os.path.exists(ref_path) and os.path.exists(cand_path)):
        print(f"[Worker {worker_id}] 文件缺失：ref={ref_path} cand={cand_path}，加入重试 idx={idx}")
        if error_idxs is not None:
            error_idxs.append(idx)
        return

    content = make_dual_image_content(prompt_text, ref_path, cand_path)
    payload = build_payload(
        model=args.model_name,
        content=content,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_tokens,
        use_openai_max_completion_tokens=args.use_openai_max_completion_tokens,
    )

    print(f"[Worker {worker_id}] pair_id={pair_id} 开始请求")
    response, status = do_one_request(payload, api_base=args.api_base, api_key=args.api_key, timeout=args.timeout)

    if status == "ok" and response:
        message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        explanation, rating = parse_llm_output(message_content or "")

        result = dict(input_row)
        result.update({
            "eval_prompt": prompt_text,
            "llm_raw_output": (message_content or "").strip(),
            "explanation": explanation,
            "rating": rating,
            "model_name": args.model_name,
        })

        with open(output_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')

        with open(index_file, 'w') as fidx:
            fidx.write(str(idx + 1))

        print(f"[Worker {worker_id}] Done idx={idx} | rating={rating}")
    else:
        print(f"[Worker {worker_id}] 请求失败，加入重试 idx={idx}, reason={status}")
        if error_idxs is not None:
            error_idxs.append(idx)

# ----------------------------
# Worker & Orchestration
# ----------------------------
def worker_fn(
    worker_id: int,
    inputs: List[dict],
    start_idx: int,
    end_idx: int,
    args: argparse.Namespace,
    index_prefix: str,
    output_prefix: str,
    prompt_text: str,
):
    index_file = f"{index_prefix}_{worker_id}.txt"
    output_file = f"{output_prefix}_{worker_id}.jsonl"

    if os.path.exists(index_file):
        try:
            with open(index_file, 'r') as f:
                persisted_idx = f.read().strip()
                if persisted_idx.isdigit():
                    start_idx = int(persisted_idx)
        except Exception:
            pass

    error_idxs: List[int] = []

    for idx in range(start_idx, end_idx):
        process_one(inputs[idx], worker_id, args, output_file, index_file, error_idxs, idx, prompt_text)

    if args.repeat_try:
        repeat_times = 0
        while repeat_times < args.max_repeat_times and error_idxs:
            print(f"[Worker {worker_id}] 重试第 {repeat_times+1} 轮，失败样本数={len(error_idxs)}")
            new_error_idxs: List[int] = []
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
    processes: List[Process] = []

    for i in range(args.num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        p = Process(
            target=worker_fn,
            args=(i, inputs, start, end, args,
                  os.path.join(need_dir_round, "data_index_worker"),
                  os.path.join(json_out_round, "eval_worker"),
                  prompt_text)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def merge_res_to_json(folder_path: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jsonl") and not filename.startswith('merged'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
    print(f"合并完成 -> {output_file}")

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
# CLI & Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate similarity for image PAIRS sourced from two folders (GT vs GEN) via LLM.")
    parser.add_argument("--api-base", type=str, default="https://api-gateway.glm.ai/v1")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model-name", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--use_openai_max_completion_tokens", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--need-dir", type=str, required=True)
    parser.add_argument("--json-output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--repeat-try", action="store_true")
    parser.add_argument("--max-repeat-times", type=int, default=1)
    parser.add_argument("--loop-until-ratio", type=float, default=0.10)
    parser.add_argument("--loop-max-times", type=int, default=1)
    parser.add_argument("--continue-from-loop", type=int, default=1)
    parser.add_argument("--test-limit", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("缺少 API Key。请使用 --api-key 或设置环境变量 API_KEY。")

    prompt_text = load_prompt(args)

    inputs, miss_gt, miss_gen = build_pairs_from_dirs(args.gt_dir, args.gen_dir)
    if args.test_limit > 0:
        inputs = inputs[: args.test_limit]

    TOTAL = len(inputs)
    if TOTAL == 0:
        print("未在两个目录中发现可配对的图片，退出。")
        safe_mkdirs(args.need_dir)
        Path(os.path.join(args.need_dir, "missing_in_gt.txt")).write_text("\n".join(miss_gt), encoding="utf-8")
        Path(os.path.join(args.need_dir, "missing_in_gen.txt")).write_text("\n".join(miss_gen), encoding="utf-8")
        return

    print(f"共配对到 {TOTAL} 对图片。未在GT中的键数：{len(miss_gt)}；未在GEN中的键数：{len(miss_gen)}")
    safe_mkdirs(args.need_dir, args.json_output_dir)

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
        if remain_ratio < args.loop_until_ratio:
            print(f"剩余比例 {remain_ratio:.4f} < {args.loop_until_ratio}，停止循环。")
            break

        run_multiprocess(cur_inputs, args, need_dir_round, json_out_round, prompt_text)

        merged_path = os.path.join(json_out_round, "merged.jsonl")
        merge_res_to_json(json_out_round, merged_path)

        done_keys = set()
        with open(merged_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                done_keys.add(obj.get("pair_id"))
        remained = [r for r in cur_inputs if r.get("pair_id") not in done_keys]
        Path(os.path.join(json_out_round, "remaining.jsonl")).write_text(
            "\n".join([json.dumps(x, ensure_ascii=False) for x in remained]), encoding="utf-8"
        )
        cur_inputs = remained
        loop_times += 1

if __name__ == "__main__":
    main()
