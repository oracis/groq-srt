# -*- coding: utf-8 -*-
"""按时间窗比较两份 SRT 的字数密度，定位"漏读"落在哪一段。

Whisper 偶尔会在块内跳读（同一块里某几句直接没输出），表现为某个时间窗
字数骤降。拿原生 SRT 当参照，差值大的窗口就是可疑漏读区。
"""
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
                r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse(path):
    """返回 [(start, end, text)]"""
    out, cur, buf = [], None, []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        s = line.strip()
        m = TS.search(s)
        if m:
            if cur:
                out.append((cur[0], cur[1], "".join(buf)))
            cur, buf = (sec(*m.group(1, 2, 3, 4)), sec(*m.group(5, 6, 7, 8))), []
        elif s and not s.isdigit() and cur:
            buf.append(s)
    if cur:
        out.append((cur[0], cur[1], "".join(buf)))
    return out


def bins(segs, width=60.0):
    n = int(max((e for _, e, _ in segs), default=0) / width) + 1
    acc = [0] * n
    for s, e, t in segs:
        acc[int(s / width)] += len(t)
    return acc


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    width = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    A, B = parse(a_path), parse(b_path)
    a, b = bins(A, width), bins(B, width)
    n = max(len(a), len(b))
    print("窗口 %ds；A=%s 共 %d 字，B=%s 共 %d 字"
          % (width, a_path, sum(a), b_path, sum(b)))
    print("%8s %6s %6s %6s  %s" % ("时间", "A", "B", "差", "提示"))
    for i in range(n):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        d = x - y
        flag = ""
        if d <= -12:
            flag = "<<< A 明显偏少（疑漏读）"
        elif d >= 12:
            flag = ">>> A 偏多"
        if abs(d) >= 12 or i % 5 == 0:
            print("%4d-%4d %6d %6d %6d  %s"
                  % (i * width, (i + 1) * width, x, y, d, flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
