# -*- coding: utf-8 -*-
"""回归：词序必须只由模型输出顺序决定，不能被时间戳排序打乱。

背景（实测121.7s 附近）：
    Whisper 原文本   他此前创作的《大英博物馆世界简史》风靡全球
    按时间戳排序后   他此前创《作的大英 博物馆简世界》史风靡 全让球
误差全部来自"每个词的时间戳独立回归、相邻倒挂"，一个字都没听错。
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import groq_srt as G

# 真实时间戳（取自 out_zhenshen_v13.words.json，已被旧代码排序污染过）
REAL = [
    ("他", 121.13, 121.31), ("此", 121.31, 121.57), ("前", 121.57, 121.77),
    ("创", 121.77, 122.11), ("作", 122.11, 122.29), ("的", 122.29, 122.41),
    ("《", 121.93, 122.45),                                  # 比"作"的 start 早
    ("大", 122.45, 122.57), ("英", 122.57, 122.69), ("博", 122.69, 122.87),
    ("物", 122.87, 123.01), ("馆", 123.01, 123.13),
    ("世", 123.13, 123.29), ("界", 123.29, 123.47), ("简", 123.03, 123.71),
    ("史", 123.71, 123.89), ("》", 123.47, 123.63),          # 比"史"的 start 晚
    ("风", 123.89, 124.17), ("靡", 124.17, 124.29), ("全", 124.29, 124.51),
    ("球", 124.51, 124.73), ("让", 124.31, 125.23),          # "让"落在"全"内部
]

TEXT = "".join(w for w, _, _ in REAL)
EXPECT = "他此前创作的《大英博物馆世界简史》风靡全球让"


def mk():
    return [{"word": w, "start": s, "end": e} for w, s, e in REAL]


def check(name, cond, extra=""):
    if cond:
        print("  [PASS] %s %s" % (name, extra))
        return 0
    print("  [FAIL] %s %s" % (name, extra))
    return 1


def main():
    bad = 0
    print("文本顺序: %s" % TEXT)
    bad += check("用例本身字序正确", TEXT == EXPECT, TEXT)

    # 1) 旧行为复现：按时间排序会打乱字序（证明这个 bug 真实存在）
    old = sorted(mk(), key=lambda w: (w["start"], w["end"]))
    old_text = "".join(w["word"] for w in old)
    print("按时间排序后: %s" % old_text)
    bad += check("旧排序确实会打乱", old_text != TEXT, old_text)

    # 2) 新行为：对齐后字序必须一字不动
    regions = [(120.9, 126.0)]
    out = G._realign_words(mk(), regions)
    new_text = "".join(w["word"] for w in out)
    bad += check("对齐后字序不变", new_text == TEXT, new_text)
    bad += check("对齐后字数不变", len(out) == len(REAL), "%d/%d" % (len(out), len(REAL)))

    # 3) 对齐后时间必须单调不回退（否则会出现负时长/闪读）
    mono = all(out[i]["end"] <= out[i + 1]["start"] + 1e-6 for i in range(len(out) - 1))
    bad += check("时间轴单调不重叠", mono,
                 "" if mono else str([(a["word"], a["end"], b["start"])
                                      for a, b in zip(out, out[1:])
                                      if a["end"] > b["start"] + 1e-6][:3]))
    bad += check("无负时长", all(w["end"] > w["start"] for w in out))

    # 4) 端到端：分段器拼回来的文本也必须一致
    cues = G._split_cues(out, max_chars=11, max_dur=4.5)
    cue_text = "".join(c["text"].replace("\n", "") for c in cues)
    bad += check("分段后文本一字不丢", cue_text == TEXT, cue_text)
    bad += check("条与条不重叠",
                 all(cues[i]["end"] <= cues[i + 1]["start"] + 1e-6
                     for i in range(len(cues) - 1)))

    # 5) 另一处实测：「可以说它是一部横跨四万年的人类精神文明成长史」
    real2 = [("可", 246.1, 246.3), ("以", 246.3, 246.5), ("说", 246.5, 246.7),
             ("它", 246.7, 246.9), ("是", 246.9, 247.1), ("一", 247.1, 247.3),
             ("部", 247.3, 247.5), ("横", 247.5, 247.7), ("跨", 247.7, 247.9),
             ("四", 248.0, 248.3), ("万", 248.3, 248.5), ("年", 248.5, 248.7),
             ("的", 248.7, 248.9), ("人", 248.62, 248.82), ("类", 248.9, 249.1),
             ("精", 249.1, 249.3), ("神", 249.3, 249.5), ("文", 249.5, 249.7),
             ("明", 249.7, 249.9), ("成", 249.9, 250.1), ("长", 250.1, 250.3),
             ("史", 250.3, 250.5)]
    t2 = "".join(w for w, _, _ in real2)
    o2 = G._realign_words([{"word": w, "start": s, "end": e} for w, s, e in real2],
                          [(246.0, 250.8)])
    bad += check("第二处字序不变", "".join(w["word"] for w in o2) == t2,
                 "".join(w["word"] for w in o2))

    print("\n%s (%d 项失败)" % ("全部通过" if bad == 0 else "有失败", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
