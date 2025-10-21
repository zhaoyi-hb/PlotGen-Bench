#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import base64
import argparse
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from multiprocessing import Process
import requests


# ----------------------------
# Utilities
# ----------------------------
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def do_one_request(
    payload: dict,
    api_base: str,
    api_key: str,
    timeout: int = 1200
) -> Tuple[Optional[dict], str]:
    """
    Simple POST wrapper. Return (json, 'ok') or (None, 'fail:xxx')
    """
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


FENCED_PY_PATTERN = re.compile(
    r"```[ \t]*([Pp]ython|py)\s*\n(.*?)\n?```",
    flags=re.DOTALL,
)


def extract_first_python_block(text: str) -> str:
    """
    Extract first ```python ...``` block content.
    Return empty string if nothing found.
    """
    matches = FENCED_PY_PATTERN.findall(text or "")
    if not matches:
        return ""
    # matches: list of tuples (lang, body)
    return matches[0][1].strip()


def safe_mkdirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ----------------------------
# Core per-item request
# ----------------------------
def build_prompt(instruct: str) -> str:
    return f"""Given the input gif, use matplotlib animation to generate an gif that is as identical as possible. Requirements:
1. Preserve the content, colors, style, layout, and details of the original gif.
2. Do not add or remove any elements.
3. Keep the resolution and aspect ratio consistent with the original gif.
4. The output should be visually almost indistinguishable from the input.
5. use ani.save("output.gif", writer="pillow", dpi=DPI) output the result gif.
6. You are only allowed to reply with code.
Do not add any explanation. 
Return the result strictly in this format:
```python
<your code>```.
"""


def build_payload(
    model: str,
    content: list,
    temperature: float,
    max_tokens: int,
    use_openai_max_completion_tokens: bool,
) -> dict:
    """
    Some gateways使用 `max_completion_tokens`，有些用 `max_tokens`。
    用开关统一一下，避免 400。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    if use_openai_max_completion_tokens:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    return payload


def process_one(
    input_row: dict,
    worker_id: int,
    args: argparse.Namespace,
    output_file: str,
    index_file: str,
    error_idxs: List[int],
    idx: int,
):
    # 构造 prompt（容错处理）
    instruct = ""
    prompt_text = build_prompt(instruct)

    # 组装内容
    content = [{"type": "text", "text": prompt_text}]

    img_path = os.path.join(args.image_root, input_row["filename"])
    img_b64 = encode_image_to_base64(img_path)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/gif;base64,{img_b64}"}
    })

    # 构造 payload
    payload = build_payload(
        model=args.model_name,
        content=content,
        temperature=args.temperature,
        max_tokens=args.max_completion_tokens,
        use_openai_max_completion_tokens=args.use_openai_max_completion_tokens,
    )

    _id = str(Path(input_row["filename"]).stem)
    print(f"[Worker {worker_id}] {_id} 开始请求")

    response, status = do_one_request(payload, api_base=args.api_base, api_key=args.api_key, timeout=args.timeout)

    if status == "ok" and response:
        # 解析模型返回
        message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        result = dict(input_row)
        result["mllm_generate"] = (message_content or "").strip()
        result["generate_prompt"] = prompt_text

        # 写结果行
        with open(output_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 更新 index
        with open(index_file, 'w') as fidx:
            fidx.write(str(idx + 1))

        # 另存代码（若能抽取到）
        code_dir = os.path.join(args.json_output_dir, "images")
        safe_mkdirs(code_dir)
        code_path = os.path.join(code_dir, f"{_id}.py")
        code = extract_first_python_block(result["mllm_generate"])
        Path(code_path).write_text(code, encoding="utf-8")

        print(f"[Worker {worker_id}] Processed index: {idx}")
    else:
        print(f"[Worker {worker_id}] 请求失败，不写入jsonl，加入重试队列 idx={idx}, reason={status}")
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
):
    index_file = f"{index_prefix}_{worker_id}.txt"
    output_file = f"{output_prefix}_{worker_id}.jsonl"

    # 断点续跑
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r') as f:
                # 若之前已经跑过一部分，从断点继续
                persisted_idx = f.read().strip()
                if persisted_idx.isdigit():
                    start_idx = int(persisted_idx)
        except Exception:
            pass

    error_idxs: List[int] = []

    for idx in range(start_idx, end_idx):
        process_one(
            inputs[idx],
            worker_id,
            args,
            output_file,
            index_file,
            error_idxs,
            idx,
        )

    if args.repeat_try:
        repeat_times = 0
        while repeat_times < args.max_repeat_times and error_idxs:
            new_error_idxs: List[int] = []
            for idx in error_idxs:
                process_one(
                    inputs[idx],
                    worker_id,
                    args,
                    output_file,
                    index_file,
                    new_error_idxs,
                    idx,
                )
            error_idxs = new_error_idxs
            repeat_times += 1


def run_multiprocess(inputs: List[dict], args: argparse.Namespace, need_dir_round: str, json_out_round: str):
    total = len(inputs)
    if total == 0:
        print("无输入数据，跳过本轮。")
        return

    # 动态计算 chunk size
    chunk_size = math.ceil(total / args.num_workers)

    processes: List[Process] = []
    for i in range(args.num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        p = Process(
            target=worker_fn,
            args=(
                i,
                inputs,
                start,
                end,
                args,
                os.path.join(need_dir_round, "data_index_worker"),
                os.path.join(need_dir_round, "generate_worker"),
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def read_json_lines(file: str) -> int:
    with open(file, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def merge_res_to_json(folder_path: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jsonl") and not filename.startswith('merged'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)


def filter_exist(input_file: str, merged_file: str, output_file: str, key_field: str = "filename"):
    # file_b 中已有的 key
    remove_keys = set()
    if os.path.exists(merged_file):
        with open(merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    remove_keys.add(obj.get(key_field))

    with open(input_file, 'r', encoding='utf-8') as fa, open(output_file, 'w', encoding='utf-8') as out:
        for line in fa:
            if line.strip():
                obj = json.loads(line)
                if obj.get(key_field) not in remove_keys:
                    out.write(json.dumps(obj, ensure_ascii=False) + '\n')


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
    print(f"合并完成！共写入 {merged_count} 条记录 -> {output_file}")


# ----------------------------
# CLI & Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multiprocess image-to-code batch runner with argparse & test-limit support."
    )
    # API & model
    parser.add_argument("--api-base", type=str, default="https://api-gateway.glm.ai/v1", help="Gateway base URL")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY", ""), help="API key (or set env API_KEY)")
    parser.add_argument("--model-name", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=1200, help="HTTP timeout seconds")
    parser.add_argument("--use_openai_max_completion_tokens", action="store_true",
                        help="If set, use 'max_completion_tokens' instead of 'max_tokens' in payload")

    # Paths & IO
    parser.add_argument("--input-file", type=str, required=True, default="metadata-with-en.jsonl", help="Input JSONL (each line a JSON object)")
    parser.add_argument("--image-root", type=str, required=True, default="/workspace/code_data/fig_to_code/images_render-human/images_render-human",help="Root dir of images")
    parser.add_argument("--need-dir", type=str, required=True, default="./static_res/",help="Working dir for each loop round (e.g., ./static_res/MODEL/ )")
    parser.add_argument("--json-output-dir", type=str, required=True, help="Output dir for merged/final JSONs")

    # Parallel & loop
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--repeat-try", action="store_true", help="Enable retry within worker loop")
    parser.add_argument("--max-repeat-times", type=int, default=1)
    parser.add_argument("--loop-until-ratio", type=float, default=0.0, help="Stop if remaining ratio < this value")
    parser.add_argument("--loop-max-times", type=int, default=3)
    parser.add_argument("--continue-from-loop", type=int, default=1, help="Resume from which loop (1 = from start)")

    # Testing
    parser.add_argument("--test-limit", type=int, default=0, help="If >0, only run the first N records of INPUT_FILE")

    return parser.parse_args()


def load_inputs(input_file: str, test_limit: int = 0) -> List[dict]:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    if test_limit and test_limit > 0:
        data = data[:test_limit]
    return data


def main():
    args = parse_args()

    if not args.api_key:
        raise SystemExit("缺少 API Key。请使用 --api-key 或设置环境变量 API_KEY。")

    # 目录准备
    safe_mkdirs(args.need_dir, args.json_output_dir)

    # 加载输入（可选仅前 N 条）
    all_inputs = load_inputs(args.input_file, args.test_limit)
    TOTAL = len(all_inputs)
    if TOTAL == 0:
        raise SystemExit("输入为空，退出。")

    print(f"当前数据量为 {TOTAL} 条（test-limit={args.test_limit}），准备并行处理。")

    # 初始化循环控制
    loop_times = args.continue_from_loop if args.continue_from_loop > 1 else 1

    # 如果从中间轮次继续，需要把输入文件切换为上一轮剩余
    current_input_file = args.input_file
    while loop_times <= args.loop_max_times:
        print(f"第 {loop_times} 次循环，模型：{args.model_name}")

        # 根据当前轮次准备本轮工作与输出目录
        need_dir_round = os.path.join(args.need_dir, str(loop_times))
        json_out_round = os.path.join(args.json_output_dir, str(loop_times))
        safe_mkdirs(need_dir_round, json_out_round)

        # 若不是第一轮，使用上一轮的 remained.jsonl 作为输入
        if loop_times > 1:
            current_input_file = os.path.join(args.json_output_dir, str(loop_times - 1), "remained.jsonl")

        # 读取本轮输入（仍然支持 test-limit，仅第一轮有意义；后续轮一般全量）
        inputs = load_inputs(current_input_file, 0 if loop_times > 1 else args.test_limit)
        cur_total = len(inputs)
        if cur_total == 0:
            print("本轮没有可处理数据，结束。")
            break

        remain_ratio = cur_total / TOTAL
        print(f"当前剩余数据量 {cur_total}，剩余比例 {remain_ratio:.4f}")
        if remain_ratio < args.loop_until_ratio:
            print(f"剩余比例 {remain_ratio:.4f} < {args.loop_until_ratio}，停止循环。")
            break

        # 并行跑本轮
        run_multiprocess(inputs, args, need_dir_round, json_out_round)

        # 合并本轮 worker 输出
        merged_path = os.path.join(json_out_round, "merged.jsonl")
        merge_res_to_json(need_dir_round, merged_path)

        # 生成 remained（用 filename 去重）
        remained_path = os.path.join(json_out_round, "remained.jsonl")
        filter_exist(current_input_file, merged_path, remained_path)

        loop_times += 1
        
        if args.test_limit > 0:
            break

    # 汇总所有轮次 merged.jsonl -> final_merged.jsonl
    final_list = [
        os.path.join(args.json_output_dir, str(i), "merged.jsonl")
        for i in range(1, loop_times)
    ]
    final_out = os.path.join(args.json_output_dir, "final_merged.jsonl")
    merge_jsonl_files(final_list, final_out)
    print(f"汇总完成：{final_out}")


if __name__ == "__main__":
    main()
