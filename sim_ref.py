# -*- coding: utf-8 -*-
"""把外部方案里的 split_subtitles_jianying_style 原样移植过来，
用我们保存的真实词表（6452 词，《众神》全片）实跑，量化它的实际表现。

只做移植 + 对比，不改它的算法逻辑，保证结论可复核。
"""
import json
import sys

import compare_drafts as cd

STRONG = {"。", "！", "？"}
WEAK = {"，", "、", "；", "："}


def clean(t):
    return t.strip().replace(" ", "")


def split_ref(words, silence_thresh=0.30, max_chars=14, min_chars=5,
              max_duration=4.5, thr=1.0, stat=None):
    """原样移植：加权打分，>= thr 才切。"""
    if not words:
        return []
    result, cur = [], []
    seg_start = words[0]["start"]
    for w in words:
        cur.append(w)
        text = "".join(clean(x["word"]) for x in cur)
        n = len(text)
        dur = w["end"] - seg_start
        score = 0.0
        if len(cur) >= 2:
            gap = w["start"] - cur[-2]["end"]
            if gap > silence_thresh:
                score += 0.6
                if stat is not None:
                    stat["silence_hit"] = stat.get("silence_hit", 0) + 1
        t = clean(w["word"])
        if t in STRONG:
            score += 0.9
            if stat is not None:
                stat["strong_hit"] = stat.get("strong_hit", 0) + 1
        elif t in WEAK:
            score += 0.35
            if stat is not None:
                stat["weak_hit"] = stat.get("weak_hit", 0) + 1
        if n >= max_chars:
            score += 999
            if stat is not None:
                stat["hard_width"] = stat.get("hard_width", 0) + 1
        elif n >= max_chars * 0.85:
            score += 0.5
            if stat is not None:
                stat["near_width"] = stat.get("near_width", 0) + 1
        if dur >= max_duration:
            score += 999
            if stat is not None:
                stat["hard_dur"] = stat.get("hard_dur", 0) + 1
        if score >= thr:
            if n < min_chars:
                continue
            result.append({"text": text.rstrip("。"), "start": seg_start, "end": w["end"]})
            cur = []
            seg_start = w["end"]
    if cur:
        text = "".join(clean(x["word"]) for x in cur)
        result.append({"text": text.rstrip("。"), "start": seg_start, "end": words[-1]["end"]})
    return result


def main():
    wp = sys.argv[1] if len(sys.argv) > 1 else "out_zhenshen_v13.words.json"
    W = json.load(open(wp, encoding="utf-8"))
    jy = cd.load_srt(r"E:/jianyin/10月23日.srt")

    print("=== 各打分项在新数据上的实际命中次数（默认参数）===")
    stat = {}
    segs = split_ref(W, stat=stat)
    for k, label in [("silence_hit", "静音 >0.30s（+0.6）"),
                     ("strong_hit", "强标点（+0.9）"),
                     ("weak_hit", "弱标点（+0.35）"),
                     ("near_width", "字数达 85%（+0.5）"),
                     ("hard_width", "字数达上限（+999）"),
                     ("hard_dur", "超时长（+999）")]:
        print("  %-22s %5d 次" % (label, stat.get(k, 0)))
    print("  -> 共切出 %d 条" % len(segs))

    print("\n=== 默认参数（max_chars=14）===")
    cues = [(s["start"], s["end"], s["text"]) for s in segs]
    cd.stats("参考方案 max_chars=14", cues)
    cd.stats("剪映原生", jy)

    print("\n=== 换用我们的 11 字上限，看是否救得回来 ===")
    for mc in (11, 10):
        s2 = split_ref(W, max_chars=mc)
        cd.stats("参考方案 max_chars=%d" % mc, [(a["start"], a["end"], a["text"]) for a in s2])

    print("\n=== 行为特征对比（关键差异）===")
    for name, cs in (("参考方案14", cues), ("剪映原生", jy)):
        gaps = [cs[i + 1][0] - cs[i][1] for i in range(len(cs) - 1)]
        gaps_sorted = sorted(gaps)
        zero = sum(1 for g in gaps if abs(g) < 1e-6)
        cover = sum(e - s for s, e, _ in cs) / cs[-1][1]
        print("%-10s 条间间隔中位 %.3fs | 零间隔占比 %.0f%% | 覆盖 %.0f%% | 最长条 %.2fs"
              % (name, gaps_sorted[len(gaps) // 2], 100.0 * zero / len(gaps),
                 100.0 * cover, max(e - s for s, e, _ in cs)))

    print("\n=== 前 6 条逐条对照 ===")
    print("  参考方案:")
    for a, b, t in cues[:6]:
        print("    %6.2f-%6.2f (%2d字) %s" % (a, b, len(t), t))
    print("  剪映原生:")
    for a, b, t in jy[:6]:
        print("    %6.2f-%6.2f (%2d字) %s" % (a, b, len(t), t))


if __name__ == "__main__":
    main()
