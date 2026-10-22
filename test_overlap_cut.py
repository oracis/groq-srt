# -*- coding: utf-8 -*-
"""验证重叠切块：split_audio 返回的 offset 必须等于块内音频的真实内容起点。

方法：造 60s 音频，每 5s 一段唯一频率的正弦（200,400,...2400Hz）。
切 4 块（重叠 3s），解码每块前 0.5s 用零交叉率估主频，对照期望档位。
同时校验每块 mvhd 时长。
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groq_srt as gs

FFMPEG = gs.find_ffmpeg()
NO_WINDOW = 0x08000000


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd)); print(r.stderr[:800]); sys.exit(1)
    return r


def estimate_freq(pcm, sr=16000):
    """零交叉率估主频（纯正弦足够精确）。"""
    crossings = 0
    for i in range(1, len(pcm)):
        if (pcm[i - 1] < 0) != (pcm[i] < 0):
            crossings += 1
    return crossings * sr / (2.0 * len(pcm))


def main():
    tmp = tempfile.mkdtemp(prefix="overlaptest-")
    src = os.path.join(tmp, "src.m4a")
    # 12 段 x 5s，频率 200..2400
    inputs = []
    for i in range(12):
        inputs += ["-f", "lavfi", "-i",
                   "sine=frequency=%d:duration=5" % (200 * (i + 1))]
    fc = "".join("[%d]" % i for i in range(12)) + "concat=n=12:v=0:a=1[out]"
    run([FFMPEG, "-y", "-v", "error"] + inputs +
        ["-filter_complex", fc, "-map", "[out]",
         "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k", src])
    total = gs.m4a_end_time(src)
    print("源时长: %.3f（期望 60.000）" % total)

    n_chunks = 4
    T = total / n_chunks
    ov = 3.0
    out_dir = os.path.join(tmp, "chunks")
    os.makedirs(out_dir, exist_ok=True)
    chunks = gs.split_audio(src, out_dir, n_chunks, FFMPEG, overlap=ov)

    print("\n块数: %d（期望 %d）" % (len(chunks), n_chunks))
    ok = True
    for i, (fp, start) in enumerate(chunks):
        exp_start = max(0.0, i * T - ov)
        exp_end = min(total, (i + 1) * T + ov)
        exp_dur = exp_end - exp_start
        # 期望首段频率: start 落在第 floor(start/5) 段
        seg_idx = int(start // 5)
        exp_freq = 200 * (seg_idx + 1)
        # 实测：解码前 0.5s 估频
        pcm_f = os.path.join(tmp, "p%03d.pcm" % i)
        run([FFMPEG, "-y", "-v", "error", "-i", fp, "-t", "0.5",
             "-f", "s16le", "-ac", "1", "-ar", "16000", pcm_f])
        import array
        with open(pcm_f, "rb") as f:
            a = array.array("h")
            a.frombytes(f.read())
        freq = estimate_freq(a)
        d = gs.m4a_end_time(fp)
        # 频率容差：档位间隔 200Hz，容差 60Hz 足够判别
        freq_ok = abs(freq - exp_freq) < 60
        dur_ok = abs(d - exp_dur) < 0.2
        start_ok = abs(start - exp_start) < 1e-6
        status = "OK " if (freq_ok and dur_ok and start_ok) else "FAIL"
        if status == "FAIL":
            ok = False
        print("[%s] 块%d: offset=%.3f(期望%.3f) 时长=%.3f(期望%.3f) "
              "首段频率=%.0f(期望%d) -> 段%d"
              % (status, i, start, exp_start, d, exp_dur, freq, exp_freq, seg_idx))

    # 覆盖性检查：所有块覆盖的并集应完整覆盖 [0, total)
    cov = []
    for i, (fp, start) in enumerate(chunks):
        s = max(0.0, i * T - ov)
        e = min(total, (i + 1) * T + ov)
        cov.append((s, e))
    covered = True
    pos = 0.0
    for s, e in sorted(cov):
        if s > pos + 0.001:
            covered = False
            print("覆盖缺口: %.3f -> %.3f" % (pos, s))
        pos = max(pos, e)
    if pos < total - 0.001:
        covered = False
        print("末尾未覆盖: %.3f < %.3f" % (pos, total))
    print("覆盖完整性: %s" % ("OK（无缺口）" if covered else "FAIL"))

    print("\n%s" % ("ALL PASS" if ok and covered else "SOME FAILED"))
    sys.exit(0 if ok and covered else 1)


if __name__ == "__main__":
    main()
