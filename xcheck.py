# -*- coding: utf-8 -*-
"""把两份 SRT 的正文做字符级对齐，量化差异并列出可疑点。

用途：拿剪映原生导出的 SRT 当"第二意见"，和新版逐字比对——
  * 复现率（一致字数占比）衡量我们离原生有多近
  * 差异片段就是"还需要人工核验的点"，可直接照单复核

注意：原生也会错，所以差异 ≠ 我们有错。这里只负责"把可疑点找全"，
不假设谁对。
"""
import difflib
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PUNCT = "，。！？；：、""''（）《》〈〉…—·,.!?;:\"'()<>[]{}%¥$#@&*+-=/\\|~^`\u3000 \t\r\n"


def read_srt_text(path):
    """取 SRT 正文，去掉序号/时间码/标点/空白，只留可比较的字。"""
    raw = open(path, encoding="utf-8-sig", errors="replace").read()
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.isdigit():
            continue
        if "-->" in s or re.match(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}", s):
            continue
        out.append(s)
    text = "".join(out)
    return "".join(ch for ch in text if ch not in PUNCT)


def diff(a, b, name_a, name_b, limit=None):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    same = sum(bl - al for tag, al, bl, _, _ in sm.get_opcodes() if tag == "equal")
    base = max(len(a), len(b), 1)
    print("【%s】%d 字   【%s】%d 字" % (name_a, len(a), name_b, len(b)))
    print("一致 %d 字，占 %.2f%%；差异片段 %d 处" % (same, 100.0 * same / base, len(ops)))
    print()
    n = 0
    for tag, al, bl, ar, br in ops:
        if limit and n >= limit:
            print("... 还有 %d 处未列出" % (len(ops) - n))
            break
        n += 1
        x, y = a[al:ar], b[bl:br]
        left_a = a[max(0, al - 8):al]
        right_a = a[ar:ar + 8]
        left_b = b[max(0, bl - 8):bl]
        right_b = b[br:br + 8]
        print("%3d. 新版: %s【%s】%s" % (n, left_a, x or "-", right_a))
        print("    原生: %s【%s】%s" % (left_b, y or "-", right_b))
    return len(ops), same, base


def main():
    if len(sys.argv) < 3:
        print("用法: python xcheck.py 新版.srt 原生.srt [--limit N]")
        return 2
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        sys.argv = [v for i, v in enumerate(sys.argv)
                    if i not in (sys.argv.index("--limit") + 1,) or v == "--limit"]
        sys.argv = [v for v in sys.argv if v != "--limit"]
    a = read_srt_text(sys.argv[1])
    b = read_srt_text(sys.argv[2])
    diff(a, b, sys.argv[1].split("/")[-1].split("\\")[-1],
         sys.argv[2].split("/")[-1].split("\\")[-1], limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
