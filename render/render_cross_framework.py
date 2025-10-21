#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Render saved matplotlib/plotly/plotnine code files into images with the same basename.
# Multi-process, bounded concurrency, per-file isolation & robust cleanup.

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ----------------------------
# Fonts setup
# ----------------------------
def _setup_chinese_fonts():
    """
    Best-effort Chinese font setup for Matplotlib and Plotly.
    """
    import os
    try:
        import matplotlib
        from matplotlib import font_manager
        import matplotlib.pyplot as plt

        # 统一 mathtext 字体 与 bbox
        plt.rcParams["mathtext.fontset"] = "dejavusans"
        plt.rcParams["savefig.bbox"] = "standard"

        env_font = os.environ.get("CHART_CN_FONT", "").strip().strip('"').strip("'")
        registered_family = None

        def _register_font_file(path):
            nonlocal registered_family
            try:
                if path and Path(path).exists():
                    font_manager.fontManager.addfont(path)
                    try:
                        fp = font_manager.FontProperties(fname=path)
                        fam = fp.get_name()
                        if fam:
                            registered_family = fam
                    except Exception:
                        pass
                    if not registered_family:
                        registered_family = Path(path).stem
                    return True
            except Exception:
                pass
            return False

        if env_font:
            _register_font_file(env_font)

        if not registered_family:
            common_paths = [
                # Windows
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\msyh.ttf",
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\msjh.ttc",
                # macOS
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/Hiragino Sans GB W3.otf",
                # Linux (Noto/Source Han)
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf",
            ]
            for p in common_paths:
                if _register_font_file(p):
                    break

        fallback_families = []
        if registered_family:
            fallback_families.append(registered_family)
        fallback_families += [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK",
            "Source Han Sans CN", "Source Han Sans", "PingFang SC",
            "Heiti SC", "WenQuanYi Micro Hei", "Arial Unicode MS"
        ]

        # 只追加 sans-serif 候选，避免强制改 family 引发度量问题
        plt.rcParams["font.sans-serif"] = fallback_families + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # --- Plotly side ---
    try:
        import plotly.io as pio
        cn_family = ", ".join([
            *( [registered_family] if 'registered_family' in locals() and registered_family else [] ),
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN",
            "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "sans-serif"
        ])
        base = pio.templates.get("plotly")
        base.layout.font.family = cn_family
        pio.templates["cn"] = base
        pio.templates.default = "cn"
    except Exception:
        pass

# ----------------------------
# 强力禁用 usetex（猴子补丁）
# ----------------------------
def _hard_disable_latex():
    """
    全局、彻底地禁用 Matplotlib 的 LaTeX 渲染：
      - 强制 rcParams['text.usetex']=False
      - Monkeypatch RcParams.__setitem__，任何将其设为 True 的尝试都会被改回 False
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # 先设为 False
    try:
        mpl.rcParams["text.usetex"] = False
    except Exception:
        pass

    # 绑定原方法
    _orig_setitem = mpl.rcParams.__class__.__setitem__

    def _guard_setitem(self, key, value):
        if key == "text.usetex" and (value is True or (isinstance(value, str) and value.lower() == "true")):
            value = False  # 强制改回 False
        return _orig_setitem(self, key, value)

    # 给 RcParams 类打猴子补丁（影响全局 rcParams）
    try:
        mpl.rcParams.__class__.__setitem__ = _guard_setitem
    except Exception:
        pass

    # 再保险：关闭 style 可能带来的 usetex
    try:
        plt.style.use('default')
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
        mpl.rcParams["savefig.bbox"] = "standard"
    except Exception:
        pass

# ----------------------------
# Safe savefig
# ----------------------------
def _safe_savefig(fig, path: Path, dpi: int, use_tight: bool = False):
    """
    安全保存图像：默认禁用 tight；必要时降 DPI；保存前再次确保禁用 usetex。
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # 多重保险：保存前强制关闭 usetex
    try:
        mpl.rcParams["text.usetex"] = False
    except Exception:
        pass

    LIM = 2**23  # Matplotlib 对单向像素的上限
    w, h = fig.get_size_inches()
    W, H = int(w * dpi), int(h * dpi)

    kw = {"dpi": dpi}
    if use_tight:
        kw["bbox_inches"] = "tight"
        kw["pad_inches"] = 0.1

    m = max(W, H)
    if m > LIM:
        scale = LIM / max(m, 1)
        kw["dpi"] = max(50, int(dpi * scale))

    try:
        fig.savefig(path, **kw)
        return
    except ValueError:
        try:
            fig.savefig(path, dpi=min(dpi, 150))
            return
        except Exception:
            fig.savefig(path, dpi=96)

# ----------------------------
# Child process runner
# ----------------------------
def _run_code_with_autosave(code_text: str, out_path: Path, dpi: int,
                            plotly_format: str, html_fallback: bool,
                            conn, keep_on_error: bool) -> None:
    """
    在隔离进程执行用户代码，并自动接管 show/save。
    - 若执行过程中出现任何异常：不产出任何文件，清理已写出的产物，并以 exitcode=1 退出。
    - 只有未出错且确有产物时才算成功。
    """
    import os, traceback, tempfile

    # 用独立 MPLCONFIGDIR，隔离系统/用户 matplotlibrc
    _tmp_mpl = tempfile.mkdtemp(prefix="mplconfig_")
    os.environ["MPLCONFIGDIR"] = _tmp_mpl

    os.environ.setdefault('MPLBACKEND', 'Agg')  # enforce non-GUI backend early

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt  # noqa

    # 先把任何 style 恢复默认
    try:
        plt.style.use('default')
    except Exception:
        pass

    # === 关键：彻底禁用 LaTeX ===
    _hard_disable_latex()

    # 然后做中文字体与其余 rc 设置
    _setup_chinese_fonts()

    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: F401
    from contextlib import suppress

    # ---- 状态变量（布尔 + Path 列表）----
    _written_files: List[Path] = []
    _saved_any: bool = False
    had_error: bool = False

    def _cleanup_outputs():
        for p in list(_written_files):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    # ---- Matplotlib patch ----
    def _mpl_autosave(*args, **kwargs):
        if not plt.get_fignums():
            plt.figure()
        fig = plt.gcf()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target = out_path.with_suffix(".png")
        _safe_savefig(fig, target, dpi, use_tight=False)
        _written_files.append(target)
        nonlocal _saved_any
        _saved_any = True

    plt.show = _mpl_autosave  # type: ignore

    # ---- Plotly patch ----
    def _patch_plotly():
        try:
            import plotly.io as pio
            from plotly.graph_objects import Figure
        except Exception:
            return

        pf = (plotly_format or "png").lower()

        def _write_plotly(fig_obj):
            nonlocal _saved_any
            try:
                if pf in {"png", "svg", "pdf"}:
                    target = out_path.with_suffix(f".{pf}")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pio.write_image(fig_obj, str(target))
                    _written_files.append(target)
                    _saved_any = True
                    return True
                elif pf == "html":
                    target = out_path.with_suffix(".html")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pio.write_html(fig_obj, file=str(target), include_plotlyjs="cdn", auto_open=False)
                    _written_files.append(target)
                    _saved_any = True
                    return True
                else:
                    target = out_path.with_suffix(".png")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pio.write_image(fig_obj, str(target))
                    _written_files.append(target)
                    _saved_any = True
                    return True
            except Exception:
                if html_fallback:
                    try:
                        target = out_path.with_suffix(".html")
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        pio.write_html(fig_obj, file=str(target), include_plotlyjs="cdn", auto_open=False)
                        _written_files.append(target)
                        _saved_any = True
                        return True
                    except Exception:
                        return False
                return False

        def _fig_show(self, *a, **k):
            _write_plotly(self)

        def _pio_show(fig, *a, **k):
            _write_plotly(fig)

        Figure.show = _fig_show  # type: ignore
        pio.show = _pio_show     # type: ignore

        with suppress(Exception):
            import plotly.offline as poff
            def _offline_plot(fig_or_data, *a, **k):
                nonlocal _saved_any
                if hasattr(fig_or_data, "to_dict"):
                    if not _write_plotly(fig_or_data) and html_fallback:
                        pass
                else:
                    target = out_path.with_suffix(".html")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pio.write_html(fig_or_data, file=str(target), include_plotlyjs="cdn", auto_open=False)
                    _written_files.append(target)
                    _saved_any = True
            poff.plot = _offline_plot  # type: ignore

    # ---- Plotnine patch ----
    def _patch_plotnine():
        try:
            import plotnine as p9
            from plotnine.ggplot import ggplot as _GG
        except Exception:
            return

        import matplotlib.pyplot as plt  # ensure plt available
        pf = (plotly_format or "png").lower()
        fmt = pf if pf in {"png", "svg", "pdf"} else "png"
        _pn_ext = f".{fmt}"

        _orig_save = getattr(_GG, "save", None)
        _orig_ggsave = getattr(p9, "ggsave", None)
        _orig_draw = getattr(_GG, "draw", None)

        def _gg_save(self, filename=None, *a, **k):
            nonlocal _saved_any
            try:
                target = out_path.with_suffix(_pn_ext)
                k = dict(k) if k else {}
                k.setdefault("dpi", dpi)
                if _orig_save:
                    ret = _orig_save(self, str(target), *a, **k)
                else:
                    fig = self.draw()
                    _safe_savefig(fig, target, dpi, use_tight=False)
                    ret = fig
                _written_files.append(target)
                _saved_any = True
                return ret
            finally:
                pass

        _GG.save = _gg_save  # type: ignore

        def _ggsave_redirect(*a, **k):
            nonlocal _saved_any
            try:
                target = out_path.with_suffix(_pn_ext)
                k = dict(k) if k else {}
                k.setdefault("dpi", dpi)
                k["filename"] = str(target)
                if _orig_ggsave:
                    ret = _orig_ggsave(*a, **k)
                else:
                    plot = k.get("plot", None)
                    if plot is None:
                        for obj in a:
                            if obj.__class__.__name__ == "ggplot":
                                plot = obj
                                break
                    if plot is None:
                        return
                    fig = plot.draw()
                    _safe_savefig(fig, target, dpi, use_tight=False)
                    ret = fig
                _written_files.append(target)
                _saved_any = True
                return ret
            finally:
                pass

        if _orig_ggsave is not None:
            p9.ggsave = _ggsave_redirect  # type: ignore

        def _gg_draw(self, *a, **k):
            nonlocal _saved_any
            fig = _orig_draw(self, *a, **k) if _orig_draw else plt.gcf()
            try:
                target = out_path.with_suffix(_pn_ext)
                _safe_savefig(fig, target, dpi, use_tight=False)
                _written_files.append(target)
                _saved_any = True
            except Exception:
                pass
            return fig

        _GG.draw = _gg_draw  # type: ignore

    # 应用补丁
    _patch_plotly()
    _patch_plotnine()

    # ---- 执行用户代码 ----
    glb = {'__name__': '__main__'}
    try:
        exec(compile(code_text, filename=str(out_path.with_suffix('.py')), mode='exec'), glb, glb)
    except Exception:
        had_error = True
        try:
            import traceback as _tb
            conn.send(("exec_error", _tb.format_exc()))
        except Exception:
            pass
    finally:
        # 仅当未出错时，才允许兜底保存
        if not had_error:
            # Matplotlib 兜底
            try:
                import matplotlib.pyplot as plt
                if (not _saved_any) and plt.get_fignums():
                    fig = plt.gcf()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    target = out_path.with_suffix(".png")
                    _safe_savefig(fig, target, dpi, use_tight=False)
                    _written_files.append(target)
                    _saved_any = True
            except Exception:
                pass

            # Plotnine 兜底
            if not _saved_any:
                try:
                    from plotnine.ggplot import ggplot as _GG
                    candidates = [v for v in glb.values() if isinstance(v, _GG)]
                    if candidates:
                        g = candidates[-1]
                        fig = g.draw()
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        target = out_path.with_suffix(".png")
                        _safe_savefig(fig, target, dpi, use_tight=False)
                        _written_files.append(target)
                        _saved_any = True
                except Exception:
                    pass

        # 成功/失败 & 清理
        try:
            if (not had_error) and _saved_any:
                conn.send(("ok", ""))
        except Exception:
            pass

        if had_error:
            # 有错误：清理所有已写出的文件（除非显式保留）
            if not keep_on_error:
                try:
                    _cleanup_outputs()
                except Exception:
                    pass
            exit_code = 1
        else:
            # 无错误：若没任何输出，也算失败（避免空文件成功）
            exit_code = 0 if _saved_any else 1

        # 清理临时 MPLCONFIGDIR
        try:
            import shutil as _sh
            _sh.rmtree(_tmp_mpl, ignore_errors=True)
        except Exception:
            pass

        import os as _os
        _os._exit(exit_code)

# ----------------------------
# Per-file orchestration (unchanged)
# ----------------------------
def _probe_outputs(out_base: Path, plotly_format: str, html_fallback: bool) -> List[Path]:
    pf = (plotly_format or "png").lower()
    if pf in {"png", "svg", "pdf"}:
        probe_exts = [f".{pf}", ".png"]
        if html_fallback:
            probe_exts.append(".html")
    elif pf == "html":
        probe_exts = [".html", ".png"]
    else:
        probe_exts = [".png", ".html"]
    return [out_base.with_suffix(ext) for ext in probe_exts]

def _cleanup_finals(out_base: Path, keep_on_error: bool):
    from contextlib import suppress
    if keep_on_error:
        return
    for ext in (".png", ".svg", ".pdf", ".html"):
        with suppress(Exception):
            f = out_base.with_suffix(ext)
            if f.exists():
                f.unlink()

def render_one(py_file: Path, dpi: int, timeout: int,
               plotly_format: str, html_fallback: bool,
               keep_on_error: bool=False) -> Tuple[Path, bool, str]:
    try:
        code_text = py_file.read_text(encoding='utf-8')
    except Exception as e:
        return py_file.with_suffix('.png'), False, f'read_error: {e}'

    out_base = py_file.with_suffix('')

    from multiprocessing import Pipe
    parent_conn, child_conn = Pipe()

    proc = mp.Process(
        target=_run_code_with_autosave,
        args=(code_text, out_base, dpi, plotly_format, html_fallback, child_conn, keep_on_error)
    )
    proc.start()

    start_time = time.time()
    child_msgs = []

    # 主动轮询，支持提前收集traceback与超时终止
    while True:
        # 收子进程消息
        try:
            while parent_conn.poll():
                child_msgs.append(parent_conn.recv())
        except Exception:
            pass

        if proc.exitcode is not None:
            break

        if (time.time() - start_time) > timeout:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(2)
            except Exception:
                pass
            _cleanup_finals(out_base, keep_on_error=False)
            extra = f" | child_msgs={child_msgs}" if child_msgs else ""
            return out_base.with_suffix(f".{(plotly_format if plotly_format!='html' else 'html')}"), False, f'timeout_after_{timeout}s{extra}'

        time.sleep(0.05)

    # 结束后检查输出
    if proc.exitcode == 0:
        for cand in _probe_outputs(out_base, plotly_format, html_fallback):
            if cand.exists() and cand.stat().st_size > 0:
                return cand, True, 'ok'
        return out_base.with_suffix(f".{(plotly_format if plotly_format!='html' else 'html')}"), False, 'no_output_found_after_success'
    else:
        _cleanup_finals(out_base, keep_on_error=False)
        extra = f" | child_msgs={child_msgs}" if child_msgs else ""
        return out_base.with_suffix(f".{(plotly_format if plotly_format!='html' else 'html')}"), False, f'child_exitcode_{proc.exitcode}{extra}'

# ----------------------------
# Batch concurrency manager
# ----------------------------
def render_many(code_files: List[Path], dpi: int, timeout: int,
                plotly_format: str, html_fallback: bool, keep_on_error: bool,
                workers: int) -> Tuple[int, int]:
    """
    受控并发执行：最多 `workers` 个子进程同时在跑，每个文件一个独立子进程。
    """
    total = len(code_files)
    ok, fail = 0, 0
    idx = 0

    # 活动子进程表：py_path -> info
    active: Dict[Path, Dict[str, Any]] = {}

    def _spawn_next():
        nonlocal idx, fail
        if idx >= total:
            return False
        py = code_files[idx]
        idx += 1

        try:
            code_text = py.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  -> FAIL: {py} (read_error: {e})")
            fail += 1
            return True

        out_base = py.with_suffix('')
        parent_conn, child_conn = mp.Pipe()

        proc = mp.Process(
            target=_run_code_with_autosave,
            args=(code_text, out_base, dpi, plotly_format, html_fallback, child_conn, keep_on_error)
        )
        proc.start()
        active[py] = {
            "proc": proc,
            "conn": parent_conn,
            "out_base": out_base,
            "start": time.time(),
            "msgs": [],
        }
        return True

    # 先填满并发槽位
    for _ in range(min(workers, total)):
        _spawn_next()

    # 主循环
    while active:
        # 轮询消息与状态
        to_remove = []
        for py, info in list(active.items()):
            proc = info["proc"]
            conn = info["conn"]
            out_base = info["out_base"]
            start = info["start"]

            # 收消息
            try:
                while conn.poll():
                    info["msgs"].append(conn.recv())
            except Exception:
                pass

            # 超时处理
            if proc.exitcode is None and (time.time() - start) > timeout:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.join(2)
                except Exception:
                    pass
                _cleanup_finals(out_base, keep_on_error=False)
                print(f'  -> FAIL: {py} (timeout_after_{timeout}s | child_msgs={info["msgs"]})')
                fail += 1
                to_remove.append(py)
                continue

            # 正常结束
            if proc.exitcode is not None:
                if proc.exitcode == 0:
                    produced = None
                    for cand in _probe_outputs(out_base, plotly_format, html_fallback):
                        if cand.exists() and cand.stat().st_size > 0:
                            produced = cand
                            break
                    if produced:
                        print(f'  -> OK: {produced}')
                        ok += 1
                    else:
                        print(f'  -> FAIL: {py} (no_output_found_after_success)')
                        fail += 1
                else:
                    _cleanup_finals(out_base, keep_on_error=False)
                    print(f'  -> FAIL: {py} (child_exitcode_{proc.exitcode} | child_msgs={info["msgs"]})')
                    fail += 1
                to_remove.append(py)

        # 清理已完成
        for py in to_remove:
            active.pop(py, None)

        # 补充新的任务
        while len(active) < workers and _spawn_next():
            pass

        # 小憩，避免忙等
        time.sleep(0.05)

    return ok, fail

# ----------------------------
# Discovery
# ----------------------------
def find_code_files(root: Path, pattern: str) -> List[Path]:
    return sorted(root.rglob(pattern))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str,
                    default=r'D:\document\fig2code\benchmark\origin_outpcut_jsons\images_render-513-564-human',
                    help='Root directory to search recursively for code files.')
    ap.add_argument('--file', type=str, help='Render a single code file (overrides --root).')
    ap.add_argument('--pattern', type=str, default='*.py', help='Glob pattern for code files.')
    ap.add_argument('--dpi', type=int, default=200, help='DPI for saved matplotlib/plotnine images.')
    ap.add_argument('--timeout', type=int, default=120, help='Per-file timeout in seconds.')
    ap.add_argument('--plotly-format', type=str, default='png',
                    choices=['png', 'svg', 'pdf', 'html'],
                    help='Export format for Plotly/Plotnine figures. Plotnine ignores html and falls back to PNG.')
    ap.add_argument('--html-fallback', action='store_true',
                    help='If static Plotly export fails (e.g., missing kaleido), fallback to HTML.')
    ap.add_argument('--keep-on-error', action='store_true',
                    help='Do not delete outputs on failure; useful for debugging.')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1),
                    help='Max concurrent rendering processes.')
    args = ap.parse_args()

    if args.file:
        code_files: List[Path] = [Path(args.file)]
    else:
        code_files = find_code_files(Path(args.root), args.pattern)

    if not code_files:
        print(f'No code files found. Root={args.root} Pattern={args.pattern}', file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(code_files)} code file(s) under {Path(args.root).resolve()} matching '{args.pattern}'.")
    print(f"[INFO] Running with up to {args.workers} concurrent worker(s).")

    # 单文件时复用原先顺序逻辑，便于 debug
    if len(code_files) == 1:
        py = code_files[0]
        print(f'[1/1] Rendering {py} ...', flush=True)
        out_path, success, msg = render_one(
            py, dpi=args.dpi, timeout=args.timeout,
            plotly_format=args.plotly_format, html_fallback=args.html_fallback,
            keep_on_error=args.keep_on_error
        )
        if success:
            print(f'  -> OK: {out_path}')
            print(f'\nDone. Success: 1, Fail: 0')
            return
        else:
            print(f'  -> FAIL: {py} ({msg})')
            print(f'\nDone. Success: 0, Fail: 1')
            sys.exit(2)

    # 多文件并发
    ok, fail = render_many(
        code_files=code_files,
        dpi=args.dpi,
        timeout=args.timeout,
        plotly_format=args.plotly_format,
        html_fallback=args.html_fallback,
        keep_on_error=args.keep_on_error,
        workers=max(1, int(args.workers)),
    )
    print(f'\nDone. Success: {ok}, Fail: {fail}')
    if fail > 0:
        sys.exit(2)

if __name__ == '__main__':
    try:
        import multiprocessing as _mp
        _mp.set_start_method('spawn', force=True)
    except Exception:
        pass
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
