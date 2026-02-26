#!/usr/bin/env python3
"""
熊猫叫声频分析工具
显示音频每一帧的能量分布和频谱
"""

import os
import sys
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_audio(input_path, output_dir=None, save_plots=True):
    """
    分析音频并显示各帧能量
    """
    print(f"正在加载音频: {input_path}")

    # 加载音频
    audio, sr = librosa.load(input_path, sr=None)
    duration = len(audio) / sr

    print(f"采样率: {sr}Hz")
    print(f"时长: {duration:.2f}秒")
    print(f"总样本数: {len(audio)}")

    # 计算帧参数
    frame_length = 2048
    hop_length = 512
    num_frames = 1 + (len(audio) - frame_length) // hop_length

    print(f"\n帧长度: {frame_length} 样本")
    print(f"hop长度: {hop_length} 样本")
    print(f"总帧数: {num_frames}")
    print(f"每帧时长: {frame_length/sr*1000:.1f}ms")

    # 计算每一帧的能量（均方根 RMS）
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.sqrt(np.mean(frames**2, axis=0))  # RMS
    frame_abs_max = np.max(np.abs(frames), axis=0)  # 最大绝对值

    # 计算每帧的中心频率
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    # 只计算有效帧数的质心
    frame_centroids = np.zeros(num_frames)
    for i in range(num_frames):
        mag = magnitude[:, i]
        if np.sum(mag) > 0:
            frame_centroids[i] = np.sum(frequencies * mag) / np.sum(mag)
        else:
            frame_centroids[i] = 0

    print("\n" + "="*60)
    print("帧能量统计 (RMS):")
    print(f"  最小值: {np.min(frame_energies):.6f}")
    print(f"  最大值: {np.max(frame_energies):.6f}")
    print(f"  平均值: {np.mean(frame_energies):.6f}")
    print(f"  中位数: {np.median(frame_energies):.6f}")

    print("\n帧振幅统计 (Max Abs):")
    print(f"  最小值: {np.min(frame_abs_max):.6f}")
    print(f"  最大值: {np.max(frame_abs_max):.6f}")
    print(f"  平均值: {np.mean(frame_abs_max):.6f}")

    # 找出能量最高的帧
    top_indices = np.argsort(frame_energies)[-10:][::-1]
    print("\n能量最高的10帧:")
    for i, idx in enumerate(top_indices):
        time_sec = idx * hop_length / sr
        print(f"  {i+1}. 帧 {idx}, 时间 {time_sec:.3f}s, 能量 {frame_energies[idx]:.6f}")

    # 计算时间轴
    time_axis = np.arange(num_frames) * hop_length / sr

    # 创建可视化
    if save_plots:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # 1. 波形图
        ax1 = axes[0]
        time_wave = np.arange(len(audio)) / sr
        ax1.plot(time_wave, audio, 'b-', linewidth=0.1)
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('振幅')
        ax1.set_title(f'波形图 - {os.path.basename(input_path)}')
        ax1.set_xlim([0, duration])
        ax1.grid(True, alpha=0.3)

        # 2. 每帧能量图
        ax2 = axes[1]
        ax2.fill_between(time_axis, 0, frame_energies, alpha=0.5, color='green')
        ax2.plot(time_axis, frame_energies, 'g-', linewidth=0.5)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('RMS 能量')
        ax2.set_title('每帧能量分布 (RMS)')
        ax2.set_xlim([0, duration])
        ax2.grid(True, alpha=0.3)

        # 3. 频谱图
        ax3 = axes[2]
        spec = librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                                 sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax3)
        ax3.set_title('频谱图 (dB)')
        plt.colorbar(spec, ax=ax3, format='%+2.0f dB')

        # 4. 帧质心频率
        ax4 = axes[3]
        ax4.plot(time_axis, frame_centroids, 'r-', linewidth=0.5)
        ax4.set_xlabel('时间 (秒)')
        ax4.set_ylabel('频率 (Hz)')
        ax4.set_title('每帧质心频率')
        ax4.set_xlim([0, duration])
        ax4.set_ylim([0, sr/2])
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n分析图已保存: {output_path}")
        plt.close()

    # 输出每帧数据到文本文件
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    csv_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_frames.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("帧号,时间(秒),RMS能量,最大振幅,质心频率(Hz)\n")
        for i in range(num_frames):
            time_sec = i * hop_length / sr
            f.write(f"{i},{time_sec:.4f},{frame_energies[i]:.6f},{frame_abs_max[i]:.6f},{frame_centroids[i]:.1f}\n")
    print(f"帧数据已保存: {csv_path}")

    return {
        'sr': sr,
        'duration': duration,
        'num_frames': num_frames,
        'frame_energies': frame_energies,
        'frame_abs_max': frame_abs_max,
        'frame_centroids': frame_centroids,
        'time_axis': time_axis
    }


def print_frame_details_by_time(input_path, start_time, end_time):
    """
    打印指定时间范围的帧详细信息
    """
    audio, sr = librosa.load(input_path, sr=None)

    frame_length = 2048
    hop_length = 512
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    duration = len(audio) / sr

    # 确保时间在有效范围内
    if end_time > duration:
        end_time = duration

    # 将时间转换为帧号
    start_frame = int(start_time * sr / hop_length)
    end_frame = int(end_time * sr / hop_length)

    print(f"\n时间 {start_time:.2f}s 到 {end_time:.2f}s 的帧详细信息:")
    print(f"帧号范围: {start_frame} 到 {end_frame-1}")
    print("-" * 70)
    print(f"{'帧号':<8}{'时间(s)':<12}{'RMS能量':<15}{'最大振幅':<15}")
    print("-" * 70)

    for i in range(start_frame, min(end_frame, num_frames)):
        start_sample = i * hop_length
        end_sample = start_sample + frame_length
        frame = audio[start_sample:end_sample]

        rms = np.sqrt(np.mean(frame**2))
        max_abs = np.max(np.abs(frame))
        time_sec = i * hop_length / sr

        print(f"{i:<8}{time_sec:<12.4f}{rms:<15.6f}{max_abs:<15.6f}")


def print_frame_details(input_path, start_frame=0, end_frame=50):
    """
    打印指定范围的帧详细信息
    """
    audio, sr = librosa.load(input_path, sr=None)

    frame_length = 2048
    hop_length = 512

    # 确保帧索引在有效范围内
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    if end_frame > num_frames:
        end_frame = num_frames

    print(f"\n帧 {start_frame} 到 {end_frame-1} 的详细信息:")
    print("-" * 70)
    print(f"{'帧号':<8}{'时间(s)':<12}{'RMS能量':<15}{'最大振幅':<15}{'样本预览':<20}")
    print("-" * 70)

    for i in range(start_frame, min(end_frame, num_frames)):
        start_sample = i * hop_length
        end_sample = start_sample + frame_length
        frame = audio[start_sample:end_sample]

        rms = np.sqrt(np.mean(frame**2))
        max_abs = np.max(np.abs(frame))
        time_sec = i * hop_length / sr

        # 样本预览（取前5个和后5个）
        preview = ""
        if len(frame) >= 10:
            preview = f"[{frame[0]:.4f}, {frame[1]:.4f}, ... {frame[-2]:.4f}, {frame[-1]:.4f}]"
        else:
            preview = str(frame[:5])

        print(f"{i:<8}{time_sec:<12.4f}{rms:<15.6f}{max_abs:<15.6f}{preview:<20}")


def compare_audio_quality(original_path, processed_path):
    """
    对比原始音频和处理后音频的帧能量
    """
    print("\n" + "="*60)
    print("音频质量对比")
    print("="*60)

    orig_info = analyze_audio(original_path, save_plots=False)
    proc_info = analyze_audio(processed_path, save_plots=False)

    print(f"\n原始音频:")
    print(f"  时长: {orig_info['duration']:.2f}秒")
    print(f"  帧数: {orig_info['num_frames']}")
    print(f"  平均能量: {np.mean(orig_info['frame_energies']):.6f}")

    print(f"\n处理后音频:")
    print(f"  时长: {proc_info['duration']:.2f}秒")
    print(f"  帧数: {proc_info['num_frames']}")
    print(f"  平均能量: {np.mean(proc_info['frame_energies']):.6f}")

    # 计算能量保留比例
    if len(orig_info['frame_energies']) > 0 and len(proc_info['frame_energies']) > 0:
        min_len = min(len(orig_info['frame_energies']), len(proc_info['frame_energies']))
        retention = np.sum(proc_info['frame_energies'][:min_len]) / (np.sum(orig_info['frame_energies'][:min_len]) + 1e-10) * 100
        print(f"\n能量保留比例: {retention:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='熊猫叫声频分析工具 - 显示每帧能量分布')
    parser.add_argument('input', help='输入音频文件')
    parser.add_argument('-o', '--output', default=None,
                       help='输出目录（默认同目录）')
    parser.add_argument('--no-plot', action='store_true',
                       help='不保存图表')
    parser.add_argument('-f', '--frames', type=str, default=None,
                       help='显示帧范围，如 0-50')
    parser.add_argument('-t', '--time', type=str, default=None,
                       help='显示时间范围，如 7-8（单位：秒）')
    parser.add_argument('--compare', type=str, default=None,
                       help='对比音频，如 --compare processed.wav')

    args = parser.parse_args()

    if args.compare:
        compare_audio_quality(args.input, args.compare)
    else:
        result = analyze_audio(args.input, args.output, save_plots=not args.no_plot)

        # 打印指定时间范围
        if args.time:
            try:
                parts = args.time.split('-')
                start = float(parts[0])
                end = float(parts[1])
                print_frame_details_by_time(args.input, start, end)
            except Exception as e:
                print(f"时间范围格式错误: {e}")
        # 打印指定帧范围
        elif args.frames:
            try:
                parts = args.frames.split('-')
                start = int(parts[0])
                end = int(parts[1])
                print_frame_details(args.input, start, end)
            except Exception as e:
                print(f"帧范围格式错误: {e}")
