# -*- coding: utf-8 -*-
"""compare_drafts.py - 对比剪映原生字幕草稿 vs 本工具生成的草稿。

剪映保存后会把 draft_content.json 加密，但同目录 `.backup/*.load.bak`
是加载时的明文快照，可以直接读。本脚本：

1. 从一个草稿目录里提取全部字幕条 (start, end, text)
2. 输出断句统计（条数/字数分布/时长分布/条间间隔）
3. 用真实音频的静音检测结果，量化每条边界与真实语音起点的误差
4. 两两对比，找出剪映的断句规律

用法:
    python compare_drafts.py <草稿名关键字1> <草稿名关键字2> ... --audio <音频>
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, r"D:/groq-srt")

DRAFT_ROOT = (r"C:/Users/DELL/AppData/Local/JianyingPro/User Data/"
              r"Projects/com.lveditor.draft/.recycle_bin")
TEXT_RE = re.compile(r'"text"\s*:\s*"(.*?)"\s*[,}]')


def find_draft(keyword):
    """在草稿库/回收站里按名字关键字找草稿目录，返回 (.json 路径, 实际目录名)。"""
    cands = []
    for root in (os.path.dirname(DRAFT_ROOT), DRAFT_ROOT):
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            if keyword in d and os.path.isdir(os.path.join(root, d)):
                cands.append(os.path.join(root, d))
    if not cands:
        return None, None
    cands.sort(key=len)
    d = cands[0]

    # 优先明文 draft_content.json，其次 .backup 里的明文快照
    p = os.path.join(d, "draft_content.json")
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                if f.read(1) == b"{":
                    return p, d
        except OSError:
            pass
    baks = sorted(glob.glob(os.path.join(d, ".backup", "*.load.bak")))
    for b in reversed(baks):
        return b, d
    return (p if os.path.exists(p) else None), d


def load_cues(path):
    """提取字幕条 [(start_sec, end_sec, text), ...]。"""
    j = json.load(open(path, encoding="utf-8"))
    texts = {m.get("id"): m for m in (j.get("materials", {}).get("texts") or [])}
    cues = []
    for tr in (j.get("tracks") or []):
        if tr.get("type") != "text":
            continue
        for seg in (tr.get("segments") or []):
            tr_ = seg.get("target_timerange") or {}
            st = (tr_.get("start") or 0) / 1e6
            du = (tr_.get("duration") or 0) / 1e6
            c = (texts.get(seg.get("material_id")) or {}).get("content") or ""
            m = TEXT_RE.search(c)
            txt = (m.group(1) if m else "").replace("\\n", " ").strip()
            if txt:
                cues.append((round(st, 3), round(st + du, 3), txt))
    cues.sort()
    return cues


def load_srt(path):
    """解析 SRT，返回 [(start, end, text), ...]。用于对比剪映导出的字幕。

    逐行扫描而非按空行切块：时间码行开新条，其后的非空行都是文本，
    遇到下一个纯数字序号（下一条的编号）则忽略。
    """
    raw = open(path, encoding="utf-8-sig", errors="replace").read()
    lines = [ln.strip() for ln in
             raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    pat = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*"
                     r"(\d+):(\d+):(\d+)[,.](\d+)")

    def secs(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    cues, cur, buf = [], None, []

    def flush():
        if cur and buf:
            txt = " ".join(buf).strip()
            if txt:
                cues.append((round(cur[0], 3), round(cur[1], 3), txt))

    for ln in lines:
        if not ln:
            continue
        m = pat.search(ln)
        if m:
            flush()
            g = m.groups()
            cur = (secs(*g[:4]), secs(*g[4:]))
            buf = []
            continue
        if cur is None:
            continue                      # 片头编号等
        if ln.isdigit() and buf:
            continue                      # 下一条的序号
        buf.append(ln)
    flush()
    cues.sort()
    return cues


def width(t):
    """显示宽度：汉字算 1，ASCII 算 0.55，空格不计。"""
    t = t.replace(" ", "")
    w = 0.0
    for ch in t:
        w += 0.55 if ord(ch) < 128 else 1.0
    return w


def stats(name, cues, regions=None):
    n = len(cues)
    if not n:
        print("%s: 无字幕" % name)
        return
    ws = [width(t) for _, _, t in cues]
    ds = [e - s for s, e, _ in cues]
    gaps = [cues[i + 1][0] - cues[i][1] for i in range(n - 1)]
    cov = sum(ds)
    span = cues[-1][1] - cues[0][0]

    print("\n" + "=" * 74)
    print("【%s】%d 条" % (name, n))
    print("=" * 74)
    print("  覆盖      : %.1fs 轴长 %.1fs（字幕占 %.0f%%，%s）"
          % (cov, span, 100 * cov / span if span else 0,
             "首尾相接无空档" if max(gaps or [0]) < 0.05 else "条间有间隔"))
    print("  单条字数  : 平均 %.1f  中位 %.0f  最长 %.0f  最短 %.0f"
          % (sum(ws) / n, sorted(ws)[n // 2], max(ws), min(ws)))
    print("  单条时长  : 平均 %.2fs 中位 %.2fs 最长 %.2fs 最短 %.2fs"
          % (sum(ds) / n, sorted(ds)[n // 2], max(ds), min(ds)))
    if gaps:
        print("  条间间隔  : 平均 %.3fs（>0.3s 的有 %d 处）"
              % (sum(gaps) / len(gaps), sum(1 for g in gaps if g > 0.3)))
    # 字数分布
    buckets = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 99)]
    print("  字数分布  : " + "  ".join(
        "%d-%d字:%d" % (a, b if b < 99 else 99, sum(1 for w in ws if a <= w <= b))
        for a, b in buckets))

    if regions:
        starts = [r[0] for r in regions]
        ends = [r[1] for r in regions]
        import bisect

        def dist(t, arr):
            i = bisect.bisect_left(arr, t)
            cand = [arr[j] for j in (i - 1, i) if 0 <= j < len(arr)]
            return min(abs(t - c) for c in cand) if cand else 999

        ds_start = [dist(s, starts) for s, _, _ in cues]
        ds_end = [dist(e, ends) for _, e, _ in cues]
        within = lambda arr, th: 100.0 * sum(1 for x in arr if x <= th) / len(arr)
        print("  边界对齐  : 起点距最近语音起点 中位 %.3fs / %.0f%% 在 0.2s 内"
              % (sorted(ds_start)[len(ds_start) // 2], within(ds_start, 0.2)))
        print("             终点距最近语音终点 中位 %.3fs / %.0f%% 在 0.2s 内"
              % (sorted(ds_end)[len(ds_end) // 2], within(ds_end, 0.2)))


def main():
    args = [a for a in sys.argv[1:]]
    audio = None
    if "--audio" in args:
        i = args.index("--audio")
        audio = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 2

    regions = None
    if audio:
        import groq_srt as gs
        ff = gs.find_ffmpeg()
        regions = gs.detect_speech_regions(audio, ff, log=print)

    loaded = []
    for kw in args:
        if os.path.isfile(kw):                       # 直接给 SRT 文件
            cues = load_srt(kw)
            stats("SRT: " + os.path.basename(kw), cues, regions)
            loaded.append((kw, cues))
            continue
        p, d = find_draft(kw)
        if not p or not os.path.exists(p):
            print("[warn] 找不到草稿/SRT: %s" % kw)
            continue
        cues = load_cues(p)
        plain = os.path.basename(p) == "draft_content.json"
        stats("%s%s" % (os.path.basename(d)[-16:],
                        "" if plain else "（.load.bak 明文快照）"), cues, regions)
        loaded.append((d, cues))

    # 两两对比：断句点是否一致
    if len(loaded) >= 2:
        (d1, c1), (d2, c2) = loaded[0], loaded[1]
        print("\n" + "=" * 74)
        print("【对比】%s  vs  %s" % (os.path.basename(d1)[-12:],
                                      os.path.basename(d2)[-12:]))
        print("=" * 74)
        same = 0
        for a, b in zip(c1, c2):
            if abs(a[0] - b[0]) < 0.15:
                same += 1
        print("  前 %d 条中起点差 <0.15s 的: %d 条（%.0f%%）"
              % (min(len(c1), len(c2)), same,
                 100.0 * same / max(1, min(len(c1), len(c2)))))
        print("\n  并排前 10 条:")
        print("  %-22s | %s" % ("A", "B"))
        for a, b in zip(c1[:10], c2[:10]):
            print("  [%6.2f-%6.2f] %-14s | [%6.2f-%6.2f] %s"
                  % (a[0], a[1], a[2][:12], b[0], b[1], b[2][:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
