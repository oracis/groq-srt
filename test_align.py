# -*- coding: utf-8 -*-
"""音频对齐测试：用已知三段语音的合成音频，验证漂移词表能被校正回气口。

复现用户看到的故障：Whisper 把词均匀摊在整段上（无间隙），
导致 3 个气口被"抹平"，字幕既不断开又压在静音上。

对 groq_srt（CLI）和 subtitle_workbench（GUI）两份实现各跑一遍，
防止两处代码走偏。
"""
import importlib
import os
import random
import struct
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, r"D:/groq-srt")
sys.path.insert(0, r"D:/subtitle-workbench")

FFMPEG = r"D:/Jianying6.01/Jianying_Pro/6.0.1.11779/ffmpeg.exe"
SR = 16000

# 真实语音区间（地面真值）：三段，两个气口
BURSTS = [(0.40, 3.20), (4.10, 6.00), (7.30, 10.50)]
TOTAL = 11.0
# 每段对应文本（正是用户截图里那 21 个字）
TEXTS = ["忠实地传递价值", "解读全球好书", "你好这里是喜马讲"]

STATE = {"ok": 0, "fail": 0}


def check(name, cond, detail=""):
    print(("  [OK] " if cond else "  [FAIL] ") + name
          + ("  " + str(detail) if not cond else ""))
    STATE["ok" if cond else "fail"] += 1


def make_audio(path_wav, path_m4a):
    rnd = random.Random(20261023)
    n = int(TOTAL * SR)
    frames = bytearray()
    for i in range(n):
        t = i / float(SR)
        in_burst = any(s <= t < e for s, e in BURSTS)
        v = int(rnd.uniform(-0.35, 0.35) * 32767) if in_burst else 0
        frames += struct.pack("<h", v)
    with wave.open(path_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", path_wav,
                    "-c:a", "aac", "-b:a", "64k", path_m4a], check=True)


def drifted_words():
    """模拟 Whisper 的漂移输出：21 个字均匀铺满 0~10.5s，词间零间隙。"""
    chars = "".join(TEXTS)
    step = TOTAL / len(chars)
    return [{"word": c, "start": round(i * step, 3),
             "end": round((i + 1) * step, 3)} for i, c in enumerate(chars)]


def run_module(name, m4a):
    """对指定模块跑全套断言。返回失败数。"""
    print("\n" + "=" * 66)
    print("模块：%s" % name)
    print("=" * 66)
    mod = importlib.import_module(name)

    # ---------- 1. 静音检测能否还原真实语音区间 ----------
    print("\n[1] detect_speech_regions 还原语音区间")
    regions = mod.detect_speech_regions(m4a, FFMPEG, log=print)
    check("探测到 3 段语音", len(regions) == 3, regions)
    for i, ((rs, re_), (bs, be)) in enumerate(zip(regions, BURSTS)):
        check("段%d 起点 %.2f≈%.2f" % (i + 1, rs, bs), abs(rs - bs) < 0.12, rs)
        check("段%d 终点 %.2f≈%.2f" % (i + 1, re_, be), abs(re_ - be) < 0.12, re_)

    # ---------- 2. 基线：不对齐时字幕什么样（复现故障） ----------
    print("\n[2] 基线（不对齐 = 修复前的行为）")
    base_cues = mod._split_cues(drifted_words(), max_chars=20)
    print("      -> %d 条: %s" % (len(base_cues), [c["text"] for c in base_cues]))
    check("漂移词表被切成 2 条（20字硬切），不是 3 条", len(base_cues) == 2,
          [c["text"] for c in base_cues])

    # ---------- 3. 对齐后：词全部落到语音区间内 ----------
    print("\n[3] _realign_words 把词钉回语音区间")
    aligned = mod._realign_words(drifted_words(), regions)
    outside = []
    for w in aligned:
        c = (w["start"] + w["end"]) / 2
        if not any(s - 0.02 <= c <= e + 0.02 for s, e in regions):
            outside.append((w["word"], round(w["start"], 2)))
    check("没有词落在静音里", not outside, outside)
    brks = [w["word"] for w in aligned if w.get("brk")]
    check("气口标记 2 处（3 段语音之间）", len(brks) == 2, brks)
    check("气口标记词正确（各段首字）", brks == ["解", "你"], brks)

    # ---------- 4. 分段结果：正好 3 条，起止贴住气口 ----------
    print("\n[4] _split_cues 得到 3 条对齐字幕")
    cues = mod._split_cues(aligned, max_chars=20)
    for c in cues:
        print("      [%05.2f - %05.2f] %s" % (c["start"], c["end"], c["text"]))
    check("切成 3 条", len(cues) == 3, [c["text"] for c in cues])
    check("文本与三句一致", [c["text"] for c in cues] == TEXTS,
          [c["text"] for c in cues])
    for i, (c, (bs, be)) in enumerate(zip(cues, BURSTS)):
        check("条%d 起点 %.2f 对齐语音 %.2f" % (i + 1, c["start"], bs),
              abs(c["start"] - bs) < 0.15, c["start"])
        check("条%d 终点 %.2f 不越入停顿（语音止 %.2f）" % (i + 1, c["end"], be),
              c["end"] <= be + mod.CUE_TAIL + 0.05, c["end"])
    check("相邻条不重叠",
          all(a["end"] <= b["start"] + 1e-6 for a, b in zip(cues, cues[1:])),
          [(a["end"], b["start"]) for a, b in zip(cues, cues[1:])])

    # ---------- 5. 端到端包装 _align_words ----------
    print("\n[5] _align_words 端到端（含日志与失败降级）")
    out = mod._align_words(drifted_words(), m4a, FFMPEG, log=print)
    check("包装层返回同样 3 段布局", sum(1 for w in out if w.get("brk")) == 2,
          sum(1 for w in out if w.get("brk")))
    check("关闭对齐时原样返回",
          mod._align_words(drifted_words(), m4a, FFMPEG, log=None,
                           enabled=False) == drifted_words())
    check("ffmpeg 缺失时原样返回",
          mod._align_words(drifted_words(), m4a, None, log=None) == drifted_words())

    # ---------- 6. 真实音频无效时的降级 ----------
    bogus = os.path.join(tempfile.gettempdir(), "_align_bogus.m4a")
    with open(bogus, "wb") as f:
        f.write(b"\x00" * 4096)
    d = drifted_words()
    check("坏音频不崩、原样返回",
          mod._align_words(d, bogus, FFMPEG, log=None) == d)
    try:
        os.remove(bogus)
    except OSError:
        pass


def main():
    tmp = tempfile.mkdtemp(prefix="aligntest-")
    wav = os.path.join(tmp, "src.wav")
    m4a = os.path.join(tmp, "src.m4a")
    make_audio(wav, m4a)
    print("合成音频: %.1fs，真实语音段 %s，中间两个气口" % (TOTAL, BURSTS))

    for name in ("groq_srt", "subtitle_workbench"):
        try:
            run_module(name, m4a)
        except Exception as e:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            STATE["fail"] += 1
            print("  [FAIL] %s 跑挂: %s" % (name, e))

    total = STATE["ok"] + STATE["fail"]
    print("\n" + "=" * 66)
    print("合计 %d 项：通过 %d，失败 %d" % (total, STATE["ok"], STATE["fail"]))
    print("ALL PASS" if STATE["fail"] == 0 else "FAILED")
    return 0 if STATE["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
