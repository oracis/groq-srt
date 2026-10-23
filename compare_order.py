# -*- coding: utf-8 -*-
"""量化"按时间戳排序"到底打乱了多少处字序。

做法：拿保序的 words，逐字比较「当前顺序」与「按(start,end)排序后的顺序」，
连续不同的位置并成一段，就是一处被排序搬家的片段。
字数完全相同，所以可以逐位比对。
"""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("words") or []
    return d


def clusters(a, b):
    """返回连续不同位置的区段 [(lo, hi)]，hi 不含。"""
    out, i, n = [], 0, len(a)
    while i < n:
        if a[i] == b[i]:
            i += 1
            continue
        j = i
        while j < n and a[j] != b[j]:
            j += 1
        # 稍微吃进一点相同字，让上下文可读
        out.append((max(0, i - 3), min(n, j + 3)))
        i = j
    return out


def main(path):
    ws = load(path)
    if not ws:
        print("没有 words")
        return 2
    order = [w["word"] for w in ws]
    srtd = [w["word"] for w in
            sorted(ws, key=lambda w: (float(w.get("start", 0)), float(w.get("end", 0))))]
    if len(order) != len(srtd):
        print("长度不一致")
        return 2

    diff_pos = sum(1 for a, b in zip(order, srtd) if a != b)
    cl = clusters(order, srtd)
    backwards = sum(1 for a, b in zip(ws, ws[1:])
                    if float(b.get("start", 0)) < float(a.get("start", 0)) - 1e-6)

    print("词数 %d，时间戳倒挂 %d 处（%.1f%%）"
          % (len(ws), backwards, 100.0 * backwards / max(len(ws) - 1, 1)))
    print("按时间排序会移动 %d 个字符位置（%.1f%%），涉及 %d 个片段"
          % (diff_pos, 100.0 * diff_pos / max(len(ws), 1), len(cl)))
    print()
    print("被打乱的片段（保序文本 -> 排序后文本）：")
    for k, (lo, hi) in enumerate(cl, 1):
        a = "".join(order[lo:hi])
        b = "".join(srtd[lo:hi])
        print("%3d. %s  ->  %s" % (k, a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "out_fixed.words.json"))
