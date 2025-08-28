#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import re
import sys
from pathlib import Path
import shutil
import tokenize

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", ".mypy_cache", ".pytest_cache",
    "build", "dist", "site-packages", "venv", ".venv", "env", ".env",
}

TARGET_DIRS = [
    Path("source/hoibench/hoibench/tasks"),
    Path("scripts/skrl"),
]

def has_chinese(text: str) -> bool:
    return bool(CJK_RE.search(text))

def strip_chinese_comments_from_bytes(content: bytes) -> tuple[bytes, int]:
    """返回 (新内容bytes, 删除的注释数量)；使用 tokenize 安全移除带中文的注释。"""
    removed = 0
    tok_iter = tokenize.tokenize(io.BytesIO(content).readline)
    kept_tokens = []
    for tok in tok_iter:
        if tok.type == tokenize.COMMENT:
            # 去掉开头井号和左右空白后判断是否含中文
            comment_body = tok.string.lstrip("#").strip()
            if has_chinese(comment_body):
                removed += 1
                continue  # 丢弃这个注释 token（行内或整行都安全）
        kept_tokens.append(tok)
    new_bytes = tokenize.untokenize(kept_tokens)
    # tokenize.untokenize 在 3.8+ 返回 bytes
    if isinstance(new_bytes, str):
        new_bytes = new_bytes.encode("utf-8")
    return new_bytes, removed

def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)

def iter_python_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*.py"):
        if p.is_symlink():
            continue
        if should_skip(p):
            continue
        yield p

def process_file(path: Path, dry_run: bool = False, backup: bool = False) -> int:
    """处理单个文件；返回删除的注释数量。"""
    try:
        original = path.read_bytes()
    except Exception as e:
        print(f"[WARN] 读取失败: {path} ({e})", file=sys.stderr)
        return 0

    try:
        new_bytes, removed = strip_chinese_comments_from_bytes(original)
    except Exception as e:
        print(f"[WARN] 解析失败: {path} ({e})", file=sys.stderr)
        return 0

    if removed > 0 and new_bytes != original:
        print(f"[MOD] {path}  删除中文注释 {removed} 处")
        if not dry_run:
            if backup and not (path.with_suffix(path.suffix + ".bak")).exists():
                try:
                    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
                except Exception as e:
                    print(f"[WARN] 备份失败: {path} ({e})", file=sys.stderr)
            try:
                path.write_bytes(new_bytes)
            except Exception as e:
                print(f"[WARN] 写回失败: {path} ({e})", file=sys.stderr)
    return removed

def main():
    parser = argparse.ArgumentParser(
        description="批量删除 Python 文件中含中文的 # 注释（保留英文注释与代码）"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="项目根目录（缺省为当前目录）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅统计/打印将要修改的内容，不写回文件"
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="为每个被修改的文件创建 .bak 备份"
    )
    args = parser.parse_args()

    # 组合目标目录
    targets = [args.root / d for d in TARGET_DIRS]

    total_files = 0
    total_removed = 0
    for root in targets:
        if not root.exists():
            print(f"[INFO] 路径不存在，跳过: {root}")
            continue
        for py in iter_python_files(root):
            total_files += 1
            total_removed += process_file(py, dry_run=args.dry_run, backup=args.backup)

    print(f"\n完成：扫描 {total_files} 个 .py 文件，删除中文注释 {total_removed} 处。")
    if args.dry_run:
        print("（dry-run 模式未写回文件）")

if __name__ == "__main__":
    main()
