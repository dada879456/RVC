#!/usr/bin/env python3
"""
对长音频进行自动分割：
- 在“有声音”的地方切分
- 去掉纯静音
- 合并间隔太短的片段
"""

import os
import argparse

import numpy as np
import librosa
import soundfile as sf


def merge_intervals(intervals, sr, min_gap_sec=0.3, min_len_sec=0.5):
    """合并间隔太短的片段，并丢弃太短的段"""
    if len(intervals) == 0:
        return []

    min_gap = int(min_gap_sec * sr)
    min_len = int(min_len_sec * sr)

    # 先合并相邻间隔较短的片段
    merged = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        gap = start - cur_end
        if gap <= min_gap:
            # 间隔太短，合并
            cur_end = end
        else:
            merged.append([cur_start, cur_end])
            cur_start, cur_end = start, end

    merged.append([cur_start, cur_end])

    # 丢弃时长太短的片段
    final = []
    for s, e in merged:
        if e - s >= min_len:
            final.append([s, e])

    return final


def extract_best_40s(
    input_path,
    output_path=None,
    top_db=30,
    target_sr=44100,
    target_duration_sec=40.0,
):
    """
    从长音频中自动截取一段约 40 秒的优质人声片段：
    - 先检测非静音区间
    - 尝试在单个最长片段中截取 40 秒
    - 若没有任何单片段 >= 40 秒，则从若干片段拼接后再截取 40 秒
    """
    print(f"加载音频用于提取优质片段: {input_path}")
    # target_sr 可以为 None，表示保持原采样率（不重采样）
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    duration = len(y) / sr
    print(f"总时长: {duration:.2f} 秒, 采样率: {sr}")

    print(f"检测非静音区间 (top_db={top_db}) ...")
    raw_intervals = librosa.effects.split(y, top_db=top_db)
    print(f"检测到 {len(raw_intervals)} 段非静音片段")

    if len(raw_intervals) == 0:
        raise RuntimeError("没有检测到任何非静音区域，无法提取优质片段。")

    target_len = int(target_duration_sec * sr)

    # 先按单个片段长度从长到短排序
    intervals_sorted = sorted(
        raw_intervals, key=lambda se: se[1] - se[0], reverse=True
    )

    # 尝试在某一个最长片段中直接截取 target_len
    best_seg = None
    for start, end in intervals_sorted:
        seg_len = end - start
        if seg_len >= target_len:
            # 在该片段中间位置截取一段 40 秒，尽量避开片段边缘可能的噪声
            offset = (seg_len - target_len) // 2
            s = start + offset
            e = s + target_len
            best_seg = y[s:e]
            print(
                f"在单个片段中截取 {target_duration_sec:.1f} 秒: "
                f"[{s/sr:.2f}s, {e/sr:.2f}s]"
            )
            break

    # 如果没有任何单片段足够长，则拼接多个片段后再截 40 秒
    if best_seg is None:
        print("没有单个片段 >= 40 秒，尝试拼接多个片段后再截取。")
        pieces = []
        total_len = 0
        for start, end in intervals_sorted:
            seg = y[start:end]
            pieces.append(seg)
            total_len += len(seg)
            if total_len >= target_len:
                break

        if total_len == 0:
            raise RuntimeError("非静音片段总长度为 0，无法提取优质片段。")

        concat = np.concatenate(pieces)
        if len(concat) <= target_len:
            best_seg = concat
            print(
                f"拼接后总长度不足 40 秒，仅导出 {len(concat)/sr:.2f} 秒的片段。"
            )
        else:
            # 从拼接后的中间位置取一段 40 秒
            mid_start = (len(concat) - target_len) // 2
            mid_end = mid_start + target_len
            best_seg = concat[mid_start:mid_end]
            print(
                f"从拼接后的中间位置截取 {target_duration_sec:.1f} 秒: "
                f"[{mid_start/sr:.2f}s, {mid_end/sr:.2f}s]"
            )

    if output_path is None:
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_best40.wav")

    sf.write(output_path, best_seg, sr)
    print(f"已导出优质 40 秒片段 -> {output_path}")


def split_audio(
    input_path,
    output_dir,
    top_db=30,
    min_gap_sec=0.3,
    min_len_sec=0.5,
    target_sr=44100,
):
    """
    自动分割音频为多个有声音的片段

    参数：
    - top_db: 非静音阈值，数值越小越“严格”，可以根据噪声情况调整
    - min_gap_sec: 如果两个片段之间间隔小于这个值，就把它们合并
    - min_len_sec: 小于这个时长的片段会被丢弃
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载音频: {input_path}")
    # target_sr 可以为 None，表示保持原采样率（不重采样）
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    duration = len(y) / sr
    print(f"总时长: {duration:.2f} 秒, 采样率: {sr}")

    # 找到所有非静音区间
    print(f"检测非静音区间 (top_db={top_db}) ...")
    raw_intervals = librosa.effects.split(y, top_db=top_db)
    print(f"初始检测到 {len(raw_intervals)} 段非静音片段")

    # 合并间隔太短、丢弃过短片段
    intervals = merge_intervals(raw_intervals, sr, min_gap_sec, min_len_sec)
    print(f"合并与筛选后剩余 {len(intervals)} 段")

    if not intervals:
        print("没有找到有效的非静音片段，请调低 top_db 再试试。")
        return

    # 逐段导出
    for idx, (start, end) in enumerate(intervals, 1):
        seg = y[start:end]
        seg_dur = len(seg) / sr
        out_name = f"seg_{idx:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, seg, sr)
        print(f"  导出第 {idx} 段: {seg_dur:.2f} 秒 -> {out_path}")

    print("分割完成！")


def main():
    parser = argparse.ArgumentParser(description="自动分割长音频为有声音的片段")
    parser.add_argument(
        "input",
        help="输入音频文件路径，例如 C:\\work\\RVC\\dataset_zhulin\\zhulin_vocal_clean.wav",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="输出目录（默认：与输入文件同目录下的 segments 子目录）",
    )
    parser.add_argument(
        "--top_db",
        type=float,
        default=30.0,
        help="非静音检测阈值，越小越严格（默认 30）",
    )
    parser.add_argument(
        "--min_gap_sec",
        type=float,
        default=0.3,
        help="小于此间隔的相邻片段会被合并（秒，默认 0.3）",
    )
    parser.add_argument(
        "--min_len_sec",
        type=float,
        default=0.5,
        help="小于此时长的片段会被丢弃（秒，默认 0.5）",
    )
    parser.add_argument(
        "--best40",
        action="store_true",
        help="不进行分段，直接从整段音频中自动提取约 40 秒优质人声到一个新文件",
    )
    parser.add_argument(
        "--keep_sr",
        action="store_true",
        help="保持原始采样率（例如输入是 48000Hz 时，不转换为 44100Hz）",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    if args.output_dir is None:
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(base_dir, f"{base_name}_segments")
    else:
        output_dir = args.output_dir

    # 根据是否保持原始采样率，决定 librosa.load 的 sr 参数
    load_sr = None if args.keep_sr else 44100

    if args.best40:
        # 只提取 40 秒优质人声，不做分段
        extract_best_40s(
            input_path=input_path,
            output_path=None,
            top_db=args.top_db,
            target_sr=load_sr,
            target_duration_sec=40.0,
        )
    else:
        split_audio(
            input_path=input_path,
            output_dir=output_dir,
            top_db=args.top_db,
            min_gap_sec=args.min_gap_sec,
            min_len_sec=args.min_len_sec,
            target_sr=load_sr,
        )


if __name__ == "__main__":
    main()