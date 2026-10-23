# -*- coding: utf-8 -*-
"""用截图里的真实文本演示分段效果：一个 30 秒长 segment -> 若干条可读字幕。"""
import sys
sys.path.insert(0, r"D:/groq-srt")
import groq_srt as gs

# 模拟 Groq 词级返回：一段 32 秒的超长 segment（这正是用户截图里"太多字挤一条"的形态）
RAW = (
    "社会礼法天子祭天地诸侯祭山川百姓敬先祖大到王室的祭祀小到民间家家户户的祭祖仪式"
    "这些仪式不断强化着尊卑长幼的伦理秩序 普通民众在日复一日的劳作中渐渐习惯了这套秩序"
    "不再追问为何生而不同 周王室用血缘与等级的双重纽带编织出一张覆盖天下的网"
)
# 按字切成"词"，每字 0.30s，句间留 0.9s 气口（> BREAK_GAP 0.6 -> 触发断开）
words, t = [], 0.0
for ch in RAW:
    if ch in " ":
        t += 0.90
        continue
    words.append({"word": ch, "start": round(t, 3), "end": round(t + 0.26, 3)})
    t += 0.30
# 句末加标点，让句末断点生效
for idx in (41, 74, len(words) - 1):
    if idx < len(words):
        words[idx]["word"] = words[idx]["word"] + "。"

print("输入：1 个 segment，宽 %.1f 字，时长 %.1f 秒" % (len(words), words[-1]["end"]))
print("旧行为（segment 级）：整段塞进 1 条字幕 ->", "这屏幕上根本放不下\n")

cues = gs._split_cues(words, max_chars=20)
print("新行为（词级 + 气口分段）：%d 条字幕\n" % len(cues))
for i, c in enumerate(cues, 1):
    dur = c["end"] - c["start"]
    print("%2d. [%05.1fs - %05.1fs] %4.1fs  %2d字  %s"
          % (i, c["start"], c["end"], dur, len(c["text"]), c["text"]))

durs = [c["end"] - c["start"] for c in cues]
widths = [len(c["text"]) for c in cues]
print("\n统计:")
print("  条数        : %d" % len(cues))
print("  单条最宽    : %d 字（上限 20）" % max(widths))
print("  单条最长    : %.1f 秒（上限 6.0）" % max(durs))
print("  单条最短    : %.1f 秒（下限 1.0）" % min(durs))
overlap = sum(1 for a, b in zip(cues, cues[1:]) if a["end"] > b["start"] + 1e-6)
print("  相邻重叠条数: %d（应为 0）" % overlap)
