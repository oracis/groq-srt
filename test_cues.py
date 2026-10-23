# -*- coding: utf-8 -*-
"""test_cues.py - 词级时间戳 -> 字幕分段 的单元测试。

构造贴近真实的中文词流（每词 2-4 字、时长 0.2-0.5s、正常语流间隔
0.02-0.15s，个别位置放 >=0.7s 静音模拟气口），验证 _split_cues 的
全部断开规则与 _merge_words 的跨块判重。
"""
import sys

sys.path.insert(0, r"D:\groq-srt")
import groq_srt as gs


def W(text, start, dur=0.3):
    return {"word": text, "start": start, "end": start + dur}


def _split_cues(words, **kw):
    """测试里的历史断言按 20 字上限设计（默认值后来对齐剪映改成了 12）。"""
    kw.setdefault("max_chars", 20)
    return gs._split_cues(words, **kw)


def texts_of(cues):
    return [c["text"] for c in cues]


def widths_of(cues):
    return [gs._display_width(c["text"]) for c in cues]


def check(name, cond, extra=""):
    print("[%s] %s%s" % ("OK" if cond else "FAIL", name,
                         "  " + str(extra) if extra else ""))
    if not cond:
        globals()["FAILED"] = True


FAILED = False

# ============ 1. 句末标点断开 ============
# 三个短句（每句 <=20 字宽），每句结尾 。 -> 应得 3 条
words = [
    W("社会", 0.0), W("礼法", 0.35), W("天子", 0.7), W("既天地", 1.05),
    W("诸侯", 1.4), W("既善川", 1.75), W("百姓", 2.1), W("敬先祖", 2.45),
    W("。", 2.8),
    W("民间", 3.2), W("家家", 3.55), W("户户", 3.9), W("的祭祖", 4.25),
    W("仪式", 4.6), W("。", 4.95),
    W("这些", 5.4), W("尊卑", 5.75), W("伦理", 6.1), W("。", 6.45),
]
cues = _split_cues(words)
check("句末标点切成 3 条", len(cues) == 3, str(texts_of(cues)))
check("第1条文本完整", cues[0]["text"] == "社会礼法天子既天地诸侯既善川百姓敬先祖。", cues[0]["text"])
check("第1条时间=首尾词(+余韵 CUE_TAIL，钳到下条前 0.02)",
      (cues[0]["start"], cues[0]["end"]) == (0.0, 3.18), str(cues[0]))
check("第3条短句也独立", cues[2]["text"] == "这些尊卑伦理。", cues[2]["text"])

# 超宽长句（28 字宽 > 20 上限）应被兜底切开，不堆成大块
long_words = [
    W("社会", 0.0), W("礼法", 0.35), W("天子", 0.7), W("既天地", 1.05),
    W("诸侯", 1.4), W("既善川", 1.75), W("百姓", 2.1), W("敬先祖", 2.45),
    W("大到", 2.8), W("王室的", 3.15), W("交四小", 3.5), W("。", 3.85),
]
cues_long = _split_cues(long_words)
check("超宽长句被切开", len(cues_long) == 2, str(texts_of(cues_long)))
check("超宽长句每块不超宽", all(w <= 20.0001 for w in widths_of(cues_long)),
      str(widths_of(cues_long)))

# ============ 2. 气口断开 ============
# 句子中间无标点，但有一个 1.2s 静音（换气）
words = [
    W("他", 0.0), W("说着", 0.3), W("就走", 0.6), W("了出去", 0.9),
    W("再也没有", 1.2), W("回头", 1.5), W("看过", 1.8), W("一眼", 2.1),   # end=2.4
    W("后来", 3.6), W("人们", 3.9), W("再也没", 4.2), W("见过他", 4.5),  # 3.6-2.4=1.2s 静音
]
cues = _split_cues(words)
check("气口切成 2 条", len(cues) == 2, str(texts_of(cues)))
check("气口断在第2条开头", cues[1]["start"] == 3.6, str(cues[1]))

# ============ 3. 短气口不断（<0.7s） ============
words = [W("啊", 0.0), W("那个", 0.35), W("我想", 0.75), W("说的是", 1.1)]
cues = _split_cues(words)
check("0.4s 间隔不断开", len(cues) == 1, str(texts_of(cues)))

# ============ 4. 逗号过半断开 ============
# 一句 26 字宽，中间有逗号：逗号处累计宽度 14 > 20*0.55=11 -> 断
words = [
    W("普通", 0.0), W("民众", 0.3), W("在日复一日", 0.6), W("的劳作", 0.95),
    W("中，", 1.25),
    W("渐渐", 1.7), W("习惯了", 2.0), W("这套", 2.3), W("秩序，", 2.6),
    W("不再", 3.0), W("追问", 3.3), W("为什么", 3.6), W("。", 3.9),
]
cues = _split_cues(words)
check("逗号处断开", len(cues) == 2, str(texts_of(cues)))
check("第1条停在第一个逗号", cues[0]["text"].endswith("中，") and "渐渐" not in cues[0]["text"], cues[0]["text"])

# ============ 5. 无标点长串 -> 逗号点/硬切兜底 ============
# 20 字宽上限，无任何标点的长词流：超过 20 必须断
words = []
t = 0.0
for i in range(12):
    words.append(W("词语%d" % i, t))
    t += 0.35
cues = _split_cues(words, max_chars=20)
check("无标点长串全部不超宽", all(w <= 20.0001 for w in widths_of(cues)), str(widths_of(cues)))
check("无标点长串切成多条", len(cues) >= 2, str(texts_of(cues)))
check("文本无丢失", "".join(texts_of(cues)) == "".join(w["word"] for w in words))

# ============ 6. 无逗点可切时硬切 ============
words = [W("甲", 0.0), W("乙", 0.3), W("丙", 0.6), W("丁", 0.9), W("戊", 1.2),
         W("己", 1.5), W("庚", 1.8), W("辛", 2.1), W("壬", 2.4), W("癸", 2.7),
         W("甲", 3.0), W("乙", 3.3), W("丙", 3.6), W("丁", 3.9), W("戊", 4.2)]
cues = _split_cues(words, max_chars=6)
check("纯单字硬切不超宽", all(w <= 6.0001 for w in widths_of(cues)), str(widths_of(cues)))
check("硬切后文本完整", "".join(texts_of(cues)) == "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊")

# ============ 7. 时长兜底 ============
# 每词 1.5s（慢速朗读），第 3 个词起累计超 7s -> 断
words = [W("慢慢的", 0.0, 1.5), W("他说", 1.6, 1.5), W("出了一", 3.2, 1.5),
         W("口气", 4.8, 1.5), W("然后", 6.4, 1.5), W("继续", 8.0, 1.5)]
cues = _split_cues(words, max_chars=100, max_dur=7.0)
check("时长上限生效", all(c["end"] - c["start"] <= 7.0 + 1e-6 for c in cues),
      str([(c["start"], c["end"]) for c in cues]))

# ============ 8. 最小时长 + 不重叠 ============
# 一个极短句（0.2s），应被拉到 >=1.0s；且不越过下一条
words = [W("好。", 0.0, 0.2), W("我们", 1.0, 0.3), W("开始。", 1.35, 0.4)]
cues = _split_cues(words)
check("最短时长 >= 1.0s", all(c["end"] - c["start"] >= 0.999 for c in cues),
      str([(c["start"], c["end"], c["text"]) for c in cues]))
check("相邻条不重叠", all(cues[i]["end"] <= cues[i + 1]["start"] - 0.019
                       for i in range(len(cues) - 1)),
      str([(c["start"], c["end"]) for c in cues]))

# ============ 9. _merge_words 跨块判重 ============
# 块A 覆盖 0-13s，块B 覆盖 10-23s：10-13s 的词两份识别
a = [W("同一", 10.0, 0.3), W("句子", 10.35, 0.3), W("被", 10.7, 0.3),
     W("转写", 11.05, 0.3), W("两次", 11.4, 0.3)]
b = [W("同一", 10.02, 0.3), W("句子", 10.38, 0.3), W("被", 10.72, 0.3),
     W("转写", 11.08, 0.3), W("两次。", 11.42, 0.3), W("后续", 11.8, 0.3)]
merged = gs._merge_words(a + b)
check("重叠词去重 11->6", len(merged) == 6, str([(w["word"], w["start"]) for w in merged]))
check("去重保留后续词", merged[-1]["word"] == "后续", merged[-1]["word"])
check("去重取时间并集", merged[0]["start"] == 10.0 and merged[0]["end"] == 10.32,
      str(merged[0]))

# 只有一块有的词要保留（漏识补缺）
c = [W("独有", 20.0, 0.3)]
merged2 = gs._merge_words(a + c)
check("独有词保留", len(merged2) == 6, str(len(merged2)))

# 标点差异不影响判重（"两次" vs "两次。"）
d = [W("同一", 30.0, 0.3), W("两次", 30.35, 0.3)]
e = [W("同一", 30.01, 0.3), W("两次。", 30.36, 0.3)]
merged3 = gs._merge_words(d + e)
check("标点差异仍判重", len(merged3) == 2, str([w["word"] for w in merged3]))

# 快速重复的同词（"的 的 的"）时间不重叠，不能误判
f = [W("的", 0.0, 0.2), W("的", 0.25, 0.2), W("的", 0.5, 0.2)]
merged4 = gs._merge_words(f)
check("连续同词不误判重", len(merged4) == 3, str(len(merged4)))

# ============ 10. _result_words 退化路径 ============
r = {"segments": [{"text": "没有词级数据", "start": 1.0, "end": 2.5}]}
w = gs._result_words(r)
check("无 words 时退化为 segment", len(w) == 1 and w[0]["word"] == "没有词级数据", str(w))
r2 = {"words": [{"word": "有", "start": 0.1, "end": 0.4}]}
w2 = gs._result_words(r2)
check("有 words 时走词级", len(w2) == 1 and w2[0]["end"] == 0.4, str(w2))

# ============ 11. 英文混排宽度 ============
words = [W("Hello", 0.0), W("world", 0.3), W("你好", 0.6), W("世界", 0.9)]
check("英文宽度 0.55", abs(gs._display_width("Hello") - 0.55 * 5) < 1e-9,
      str(gs._display_width("Hello")))
check("中文宽度 1", gs._display_width("你好") == 2.0)

# ============ 12. 默认单行字数对齐剪映（12 字） ============
check("默认单行上限 = 11（离线扫描 10~14 档，11 与原生 703 条最吻合）", gs.CUE_MAX_CHARS == 11,
      str(gs.CUE_MAX_CHARS))

# ============ 13. 巨型词二次切分（Groq 不返回 words 的退化路径） ============
# 实测旧草稿里有 223 字 / 59 秒的单条字幕，就是这里没切
monster = ("社会礼法天子祭天地诸侯祭山川百姓敬先祖大到王室的祭祀小到民间"
           "家家户户的祭祖仪式这些仪式不断强化着尊卑长幼的伦理秩序普通民众"
           "在日复一日的劳作中渐渐习惯了这套秩序不再追问为何生而不同周王室"
           "用血缘与等级的双重纽带编织出一张覆盖天下的网")
mw = gs._display_width(monster)
exploded = gs._explode_long_words([W(monster, 10.0, dur=59.0)], gs.CUE_MAX_CHARS)
check("巨型词被切成多片", len(exploded) > 5, str(len(exploded)))
check("无一片超过 1.6 倍上限",
      all(gs._display_width(e["word"]) <= gs.CUE_MAX_CHARS * 1.6 for e in exploded),
      [gs._display_width(e["word"]) for e in exploded])
check("切分后文本拼回原文", "".join(e["word"] for e in exploded) == monster)
check("时间轴首尾保持", (exploded[0]["start"], exploded[-1]["end"]) == (10.0, 69.0),
      (exploded[0]["start"], exploded[-1]["end"]))
check("时间单调递增",
      all(a["end"] <= b["start"] + 1e-6 for a, b in zip(exploded, exploded[1:])))
check("总时长不变",
      abs(sum(e["end"] - e["start"] for e in exploded) - 59.0) < 0.05,
      str(sum(e["end"] - e["start"] for e in exploded)))

# 走完整分段：不会再出现几十秒的巨型字幕
# 上限取 CUE_MAX_DUR * 1.25：断点在标点上时会放宽到 1.25 倍宽度，
# 单个超宽"词"的时间片也随之到 6.7s 左右（原生最长 4.4s，这里只防巨型条）
cues_monster = gs._split_cues([W(monster, 10.0, dur=59.0)])
check("巨型词经分段后无超长条",
      all(c["end"] - c["start"] <= gs.CUE_MAX_DUR * 1.25 + 0.5 for c in cues_monster),
      [(round(c["end"] - c["start"], 1), len(c["text"])) for c in cues_monster])
check("巨型词经分段后无超宽条",
      all(gs._display_width(c["text"]) <= gs.CUE_MAX_CHARS for c in cues_monster))

# 正常短词不受影响
check("正常词不被二次切分",
      len(gs._explode_long_words([W("你好世界", 0.0)], 12)) == 1)

# 回归：巨型词二次切分后，各片 + 余韵 CUE_TAIL 曾互相压住
# （实测《众神》464.4s 处三条 0.2s 碎条重叠，剪映里是肉眼可见的糊字）
_long = W("让人在精神自救的人的力量与气魄架接到人的身体之上在精神世界里", 464.40, 0.90)
_next = W("继续", 465.30, 0.30)
_ov = gs._split_cues([_long, _next], max_chars=10)
check("巨型词切分后各条不重叠",
      all(_ov[i]["end"] <= _ov[i + 1]["start"] + 1e-9 for i in range(len(_ov) - 1)),
      [(round(c["start"], 2), round(c["end"], 2)) for c in _ov])
check("巨型词切分后不丢字",
      "".join(c["text"] for c in _ov) == "让人在精神自救的人的力量与气魄架接到人的身体之上在精神世界里继续",
      texts_of(_ov))

print()
if FAILED:
    print("=== 存在失败项 ===")
    sys.exit(1)
print("=== ALL PASS ===")
