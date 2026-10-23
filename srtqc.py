# -*- coding: utf-8 -*-
"""SRT 质量体检：条数/字数/时长分布 + 重叠/闪读/超长等硬伤。"""
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
                r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse(path):
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


def report(path):
    segs = parse(path)
    n = len(segs)
    chars = [len(t) for _, _, t in segs]
    durs = [e - s for s, e, _ in segs]
    avg_c = sum(chars) / n
    avg_d = sum(durs) / n
    over = [(i, segs[i][0], segs[i][1]) for i in range(n - 1)
            if segs[i][1] > segs[i + 1][0] + 1e-6]
    neg = [(i, d) for i, d in enumerate(durs) if d <= 0.05]
    flash = [(i, round(d, 3)) for i, d in enumerate(durs) if 0.05 < d < 0.35]
    longc = [(i, c) for i, c in enumerate(chars) if c > 16]
    longd = [(i, round(d, 2)) for i, d in enumerate(durs) if d > 5.0]
    print("%s" % path)
    print("  条数 %d | 均 %.1f 字 | 均 %.2f 秒 | 最长 %.2f 秒/%d 字"
          % (n, avg_c, avg_d, max(durs), chars[durs.index(max(durs))]))
    print("  重叠 %d 条 | 零时长 %d 条 | 闪读(<0.35s) %d 条 | >16字 %d 条 | >5秒 %d 条"
          % (len(over), len(neg), len(flash), len(longc), len(longd)))
    if over[:3]:
        print("  重叠示例: %s" % over[:3])
    if flash[:5]:
        print("  闪读示例: %s" % flash[:5])
    if longc[:5]:
        print("  超长示例: %s" % longc[:5])
    return segs


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
        print()
