# -*- coding: utf-8 -*-
"""replay_words.py - 用落盘的词级时间戳离线重放断句，调参不必再调 API。

先跑一次带 --dump-words 的转写拿到 <out>.words.json，之后改任何断句参数
（单行字数、气口门槛、软断点…）都只需秒级重放即可看结果。

    python replay_words.py out.words.json out.srt --max-chars 11
    python replay_words.py out.words.json --sweep 10,11,12,13,14
"""
import argparse
import json
import sys

import groq_srt as gs


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_srt(cues, path):
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append("%d\n%s --> %s\n%s\n"
                      % (i, gs.srt_ts(c["start"]), gs.srt_ts(c["end"]), c["text"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    return len(blocks)


def profile(cues):
    w = [gs._display_width(c["text"]) for c in cues]
    d = [c["end"] - c["start"] for c in cues]
    w_sorted = sorted(w)
    return dict(n=len(cues), avg_w=sum(w) / len(w), med_w=w_sorted[len(w) // 2],
                avg_d=sum(d) / len(d), max_w=max(w))


def main():
    ap = argparse.ArgumentParser(description="用词级时间戳离线重放断句")
    ap.add_argument("words", help="--dump-words 产出的 .words.json")
    ap.add_argument("out", nargs="?", default=None, help="输出 SRT（--sweep 时可省）")
    ap.add_argument("--max-chars", type=int, default=gs.CUE_MAX_CHARS)
    ap.add_argument("--max-dur", type=float, default=gs.CUE_MAX_DUR)
    ap.add_argument("--sweep", default=None,
                    help="扫描多档单行字数，逗号分隔，如 10,11,12,13")
    a = ap.parse_args()

    words = load_words(a.words)
    print("[..] 载入 %d 个词" % len(words))

    if a.sweep:
        print("%-5s %6s %8s %7s %8s %6s" % ("上限", "条数", "均字数", "中位", "均时长", "最宽"))
        for mc in [int(x) for x in a.sweep.split(",")]:
            p = profile(gs._split_cues(words, mc, a.max_dur))
            print("%-5d %6d %8.1f %7d %8.2f %6d"
                  % (mc, p["n"], p["avg_w"], p["med_w"], p["avg_d"], p["max_w"]))
        return 0

    if not a.out:
        ap.error("需要输出路径（或用 --sweep）")
    cues = gs._split_cues(words, a.max_chars, a.max_dur)
    n = write_srt(cues, a.out)
    p = profile(cues)
    print("[ok] %s：%d 条，均 %.1f 字 / %.2fs，中位 %d 字，最宽 %d"
          % (a.out, n, p["avg_w"], p["avg_d"], p["med_w"], p["max_w"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
