#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量/单个渲染 GIF（保存到源文件同目录）：
- 遍历目录中的 *.py（默认 ./images）或仅运行单个文件（--file）
- 将 ani.save("output.gif", ...) 替换为 ani.save("<同目录>/<编号>.gif", ...)
- 支持跳过已有 GIF 或覆盖
- 支持渲染超时（默认 180 秒）
- 单个脚本失败不会中断，最后汇总失败列表

用法：
  # 批量（默认 images 目录）
  python batch_render_gifs.py

  # 指定目录
  python batch_render_gifs.py --src images

  # 递归子目录
  python batch_render_gifs.py --recursive

  # 直接改源文件
  python batch_render_gifs.py --inplace

  # 指定解释器
  python batch_render_gifs.py --python python3

  # 只运行单个文件（优先于 --src）
  python batch_render_gifs.py --file D:/path/to/123.py

  # 跳过已有 GIF
  python batch_render_gifs.py --skip-existing

  # 自定义渲染超时（秒）
  python batch_render_gifs.py --timeout 120
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import List

SAVE_CALL_PATTERN = re.compile(
    r'''
    ani\.save           # ani.save
    \s*\(               # (
    \s*(['"])output\.gif\1   # "output.gif" 或 'output.gif'
    \s*,                # 后面紧跟逗号
    ''',
    re.VERBOSE
)

def pick_gif_path(py_path: Path) -> str:
    stem = py_path.stem
    gif_name = stem if stem.endswith(".gif") else f"{stem}.gif"
    return (py_path.parent / gif_name).as_posix()

def rewrite_save_call(code: str, gif_fullpath: str) -> str:
    return SAVE_CALL_PATTERN.sub(f'ani.save("{gif_fullpath}",', code)

def run_one_file(py_to_run: Path, python_bin: str, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [python_bin, str(py_to_run)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            args=e.cmd,
            returncode=-1,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + f"\n❌ 超时（超过 {timeout} 秒）"
        )

def process_file(py_path: Path, inplace: bool, python_bin: str, timeout: int, skip_existing: bool) -> dict:
    gif_path = pick_gif_path(py_path)
    if skip_existing and Path(gif_path).exists():
        return {
            "file": py_path.name,
            "gif": gif_path,
            "returncode": 0,
            "stdout": "",
            "stderr": f"⚠️ 已存在，跳过渲染",
            "replaced": False,
            "skipped": True
        }

    original = py_path.read_text(encoding="utf-8")
    modified = rewrite_save_call(original, gif_path)
    replaced = (modified != original)

    if inplace:
        py_path.write_text(modified, encoding="utf-8")
        to_run = py_path
        tmp_created = False
    else:
        to_run = py_path.parent / f"_tmp_{py_path.name}"
        to_run.write_text(modified, encoding="utf-8")
        tmp_created = True

    cp = run_one_file(to_run, python_bin, timeout)

    if tmp_created:
        try:
            to_run.unlink()
        except Exception:
            pass

    return {
        "file": py_path.name,
        "gif": gif_path,
        "returncode": cp.returncode,
        "stdout": cp.stdout,
        "stderr": cp.stderr,
        "replaced": replaced,
        "skipped": False
    }

def iter_py_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.py" if recursive else "*.py"
    return sorted(root.glob(pattern))

def run_single_file(py_file: Path, inplace: bool, python_bin: str, timeout: int, skip_existing: bool) -> None:
    if not py_file.exists():
        print(f"❌ 文件不存在：{py_file}")
        return
    if py_file.suffix.lower() != ".py":
        print(f"❌ 不是 .py 文件：{py_file}")
        return

    print(f"单文件模式：处理 {py_file} …")
    res = process_file(py_file, inplace, python_bin, timeout, skip_existing)

    if res.get("skipped"):
        print(f"   ⚠️ 跳过渲染（已存在 GIF）→ {res['gif']}")
        return

    if not res["replaced"]:
        print(f"   ⚠️ {py_file.name} 中未发现 ani.save(\"output.gif\", …)，已按原样运行。")

    if res["returncode"] == 0:
        print(f"   ✅ 渲染完成 → {res['gif']}")
    else:
        print(f"   ❌ 渲染失败（退出码 {res['returncode']}）：{py_file.name}")
        if res["stderr"].strip():
            print("   └─ 错误摘要（前10行）：")
            print("\n".join("      " + line for line in res["stderr"].splitlines()[:10]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(r"./images"),
                    help="包含待运行 .py 的目录（默认 ./images）")
    ap.add_argument("--file", "-f", type=Path, help="仅运行指定的单个 .py 文件（优先级高于 --src）")
    ap.add_argument("--recursive", action="store_true", help="递归子目录（仅批量模式）")
    ap.add_argument("--inplace", action="store_true", help="直接改源文件并运行（默认用临时文件，不改源码）")
    ap.add_argument("--python", dest="python_bin", default="python", help="Python 解释器（如 python3）")
    ap.add_argument("--skip-existing", action="store_true", help="如果同名 GIF 已存在则跳过渲染")
    ap.add_argument("--timeout", type=int, default=180, help="渲染超时秒数（默认 180 秒）")
    args = ap.parse_args()

    if args.file:
        run_single_file(args.file, args.inplace, args.python_bin, args.timeout, args.skip_existing)
        return

    if not args.src.exists():
        print(f"❌ 目录不存在：{args.src}")
        return

    py_files = iter_py_files(args.src, args.recursive)
    if not py_files:
        print(f"⚠️ 没找到 .py 文件：{args.src}")
        return

    print(f"共找到 {len(py_files)} 个脚本，开始渲染（单个失败不影响后续）…\n")

    failures = []
    for i, py_path in enumerate(py_files, 1):
        print(f"[{i}/{len(py_files)}] 处理 {py_path} …")
        res = process_file(py_path, args.inplace, args.python_bin, args.timeout, args.skip_existing)

        if res.get("skipped"):
            print(f"   ⚠️ 跳过渲染（已存在 GIF）→ {res['gif']}")
            continue

        if not res["replaced"]:
            print(f"   ⚠️ {py_path.name} 中未发现 ani.save(\"output.gif\", …)，已按原样运行。")

        if res["returncode"] == 0:
            print(f"   ✅ 渲染完成 → {res['gif']}")
        else:
            print(f"   ❌ 渲染失败（退出码 {res['returncode']}），已跳过继续：{py_path.name}")
            if res["stderr"].strip():
                print("   └─ 错误摘要（前10行）：")
                print("\n".join("      " + line for line in res["stderr"].splitlines()[:10]))
            failures.append(py_path.name)

    print("\n—— 批量任务完成 ——")
    if failures:
        print(f"❌ 失败 {len(failures)} 个：{failures}")
    else:
        print("✅ 全部成功")

if __name__ == "__main__":
    main()
