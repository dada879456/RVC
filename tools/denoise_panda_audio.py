#!/usr/bin/env python3
"""
熊猫叫声音频降噪工具
使用多种滤波器组合进行背景噪音去除，保留熊猫叫声
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def band_stop_filter(audio, sr, low_freq=55, high_freq=65):
    """
    带阻滤波器
    去除特定频率的噪音（如工频干扰 50Hz/60Hz）
    """
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='bandstop')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio


def amplitude_based_denoise(audio, sr, frame_length=2048, hop_length=512, threshold=0.01):
    """
    基于振幅阈值的降噪
    振幅低于阈值的帧被视为噪音并降噪
    振幅高于阈值的帧（熊猫叫声）完全保留
    """
    # 计算每帧的最大振幅
    frame_max_amps = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i+frame_length]
        frame_max_amps.append(np.max(np.abs(frame)))
    frame_max_amps = np.array(frame_max_amps)
    num_frames = len(frame_max_amps)

    # 创建掩码：True = 保留，False = 降噪
    # 振幅高于阈值的帧保留，低于阈值的帧降噪
    preserve_mask = frame_max_amps >= threshold

    print(f"  振幅阈值: {threshold}")
    print(f"  总帧数: {num_frames}")
    print(f"  保留帧数: {np.sum(preserve_mask)} ({np.sum(preserve_mask)/num_frames*100:.1f}%)")
    print(f"  降噪帧数: {np.sum(~preserve_mask)} ({np.sum(~preserve_mask)/num_frames*100:.1f}%)")

    # 创建时域掩码
    # 每帧应用平滑过渡，避免爆破音
    mask = np.ones(len(audio))

    # 对需要降噪的帧，应用衰减
    for i, preserve in enumerate(preserve_mask):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        if not preserve:
            # 噪音帧：衰减 70%
            mask[start:end] = 0.3

    # 应用掩码
    clean_audio = audio * mask

    return clean_audio


def high_pass_filter(audio, sr, cutoff=20):
    """
    高通滤波器
    去除超低频噪音，保留熊猫叫声
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    # 使用更平缓的滤波器过渡
    b, a = signal.butter(2, normalized_cutoff, btype='high')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio


def low_pass_filter(audio, sr, cutoff=15000):
    """
    低通滤波器
    去除高频噪音
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(2, normalized_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio


def notch_filter(audio, sr, frequencies=[50, 60, 100, 120, 150, 200]):
    """
    多 notch 滤波器
    去除特定频率的干扰
    """
    for freq in frequencies:
        nyquist = sr / 2
        low = (freq - 2) / nyquist
        high = (freq + 2) / nyquist
        if 0 < low < high < 1:
            b, a = signal.butter(4, [low, high], btype='bandstop')
            audio = signal.filtfilt(b, a, audio)
    return audio


def normalize_audio(audio, target_db=-6):
    """
    音频标准化
    """
    # 转换为dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
    # 计算平均dB
    mean_db = np.mean(audio_db)
    # 计算增益
    gain = target_db - mean_db
    # 应用增益
    audio_normalized = audio * (10 ** (gain / 20))
    # 防止削波
    max_val = np.max(np.abs(audio_normalized))
    if max_val > 0.95:
        audio_normalized = audio_normalized * 0.95 / max_val
    return audio_normalized


def remove_silence_gentle(audio, sr, threshold_db=-50, min_silence_len=200):
    """
    温和去除静音部分
    使用更低的阈值，避免去除轻柔的熊猫叫声
    """
    # 转换为dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
    # 创建掩码
    mask = audio_db > threshold_db
    # 只去除很长的静音片段
    mask = signal.medfilt(mask.astype(float), kernel_size=min_silence_len // 20 + 1)
    # 转换为布尔数组用于索引
    mask = mask > 0
    # 应用掩码
    if np.sum(mask) > 0:
        audio_trimmed = audio[mask]
    else:
        audio_trimmed = audio
    return audio_trimmed


def process_panda_audio_filter_only(input_path, output_path, sr=44100):
    """
    仅使用滤波器的降噪模式
    完全不进行频谱处理，只用滤波器
    适合保留完整叫声
    """
    print(f"正在加载音频: {input_path}")
    audio, orig_sr = librosa.load(input_path, sr=sr)

    original_len = len(audio)
    print(f"音频时长: {original_len/sr:.2f}秒")
    print(f"采样率: {orig_sr}Hz")

    # 步骤1: 去除静音（温和）
    print("步骤1: 温和去除静音部分...")
    audio = remove_silence_gentle(audio, sr)

    # 步骤2: 超高通滤波（只去除人耳听不到的超低频）
    print("步骤2: 应用超高通滤波...")
    audio = high_pass_filter(audio, sr, cutoff=20)

    # 步骤3: notch 滤波器去除特定频率干扰
    print("步骤3: 应用notch滤波器去除干扰...")
    audio = notch_filter(audio, sr)

    # 步骤4: 带阻滤波器
    print("步骤4: 应用带阻滤波器...")
    audio = band_stop_filter(audio, sr, low_freq=55, high_freq=65)

    # 步骤5: 低通滤波
    print("步骤5: 应用低通滤波...")
    audio = low_pass_filter(audio, sr, cutoff=15000)

    # 步骤6: 标准化音量
    print("步骤6: 标准化音量...")
    audio = normalize_audio(audio, target_db=-6)

    # 再次温和去除静音
    print("步骤7: 最终去除静音...")
    audio = remove_silence_gentle(audio, sr)

    # 保存结果
    print(f"正在保存: {output_path}")
    sf.write(output_path, audio, sr)

    print(f"处理完成！")
    print(f"原始时长: {original_len/sr:.2f}秒")
    print(f"处理后时长: {len(audio)/sr:.2f}秒")
    print(f"保留比例: {len(audio)/original_len*100:.1f}%")


def process_panda_audio(input_path, output_path, method='filter_only',
                        normalize=True, remove_sil=True, high_pass=True,
                        sr=44100, amplitude_threshold=0.01):
    """
    处理熊猫叫声音频

    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        method: 降噪方法
            - 'filter_only': 仅滤波（推荐，用于保留叫声）
            - 'amplitude_threshold': 基于振幅阈值（振幅>=阈值保留，<阈值降噪）
        normalize: 是否标准化音量
        remove_sil: 是否去除静音
        high_pass: 是否使用高通滤波
        amplitude_threshold: 振幅阈值，低于此值的帧被视为噪音
    """
    # 如果是 filter_only 模式，使用专用函数
    if method == 'filter_only':
        process_panda_audio_filter_only(input_path, output_path, sr)
        return

    # 如果是 amplitude_threshold 模式
    if method == 'amplitude_threshold':
        print(f"正在加载音频: {input_path}")
        audio, orig_sr = librosa.load(input_path, sr=sr)

        print(f"音频时长: {len(audio)/sr:.2f}秒")
        print(f"采样率: {orig_sr}Hz")

        # 步骤1: 高通滤波
        if high_pass:
            print("步骤1: 应用高通滤波...")
            audio = high_pass_filter(audio, sr, cutoff=30)

        # 步骤2: 基于振幅阈值的降噪
        print(f"步骤2: 应用振幅阈值降噪（阈值={amplitude_threshold}）...")
        audio = amplitude_based_denoise(audio, sr, threshold=amplitude_threshold)

        # 步骤3: 低通滤波
        print("步骤3: 应用低通滤波...")
        audio = low_pass_filter(audio, sr, cutoff=15000)

        # 步骤4: 标准化音量
        if normalize:
            print("步骤4: 标准化音量...")
            audio = normalize_audio(audio, target_db=-6)

        # 步骤5: 去除静音
        if remove_sil:
            print("步骤5: 去除静音...")
            audio = remove_silence_gentle(audio, sr)

        # 保存结果
        print(f"正在保存: {output_path}")
        sf.write(output_path, audio, sr)

        print(f"处理完成！")
        print(f"原始时长: {len(librosa.load(input_path, sr=sr)[0])/sr:.2f}秒")
        print(f"处理后时长: {len(audio)/sr:.2f}秒")
        return

    print(f"正在加载音频: {input_path}")
    audio, orig_sr = librosa.load(input_path, sr=sr)

    print(f"音频时长: {len(audio)/sr:.2f}秒")
    print(f"采样率: {orig_sr}Hz")

    # 步骤1: 去除静音
    if remove_sil:
        print("步骤1: 去除静音部分...")
        audio = remove_silence_gentle(audio, sr)

    # 步骤2: 高通滤波
    if high_pass:
        print("步骤2: 应用高通滤波...")
        audio = high_pass_filter(audio, sr, cutoff=30)

    # 步骤3: 降噪处理
    print(f"步骤3: 应用{method}降噪...")
    if method == 'amplitude_threshold':
        audio = amplitude_based_denoise(audio, sr, threshold=amplitude_threshold)
    else:
        print(f"警告: 未知方法 {method}")

    # 步骤4: 低通滤波
    print("步骤4: 应用低通滤波...")
    audio = low_pass_filter(audio, sr, cutoff=15000)

    # 步骤5: 标准化音量
    if normalize:
        print("步骤5: 标准化音量...")
        audio = normalize_audio(audio, target_db=-6)

    # 去除处理后的静音
    if remove_sil:
        print("步骤6: 再次去除静音...")
        audio = remove_silence_gentle(audio, sr)

    # 保存结果
    print(f"正在保存: {output_path}")
    sf.write(output_path, audio, sr)

    print(f"处理完成！")
    print(f"原始时长: {len(librosa.load(input_path, sr=sr)[0])/sr:.2f}秒")
    print(f"处理后时长: {len(audio)/sr:.2f}秒")


def batch_process(input_dir, output_dir, method='filter_only', **kwargs):
    """
    批量处理音频文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']

    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in supported_formats]

    print(f"找到 {len(files)} 个音频文件")

    for i, file in enumerate(files, 1):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir,
                                    f"clean_{os.path.splitext(file)[0]}.wav")
        print(f"\n[{i}/{len(files)}] 正在处理: {file}")
        process_panda_audio(input_path, output_path, method, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='熊猫叫声音频降噪工具 - 保留叫声专用')
    parser.add_argument('input', help='输入音频文件或目录')
    parser.add_argument('-o', '--output', default=None,
                       help='输出文件或目录')
    parser.add_argument('-m', '--method', default='filter_only',
                       choices=['filter_only', 'amplitude_threshold'],
                       help='降噪方法: filter_only(仅滤波，推荐保留叫声), amplitude_threshold(基于振幅阈值)')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='振幅阈值：振幅>=阈值保留，<阈值降噪 (默认: 0.01)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='不标准化音量')
    parser.add_argument('--no-remove-silence', action='store_true',
                       help='不去除静音')
    parser.add_argument('--no-high-pass', action='store_true',
                       help='不使用高通滤波')
    parser.add_argument('--sr', type=int, default=44100,
                       help='采样率')

    args = parser.parse_args()

    if os.path.isfile(args.input):
        if args.output is None:
            output_path = os.path.splitext(args.input)[0] + '_clean.wav'
        else:
            output_path = args.output

        process_panda_audio(
            args.input, output_path,
            method=args.method,
            normalize=not args.no_normalize,
            remove_sil=not args.no_remove_silence,
            high_pass=not args.no_high_pass,
            sr=args.sr,
            amplitude_threshold=args.threshold
        )
    elif os.path.isdir(args.input):
        if args.output is None:
            output_dir = os.path.join(args.input, 'cleaned')
        else:
            output_dir = args.output

        batch_process(
            args.input, output_dir,
            method=args.method,
            normalize=not args.no_normalize,
            remove_sil=not args.no_remove_silence,
            high_pass=not args.no_high_pass,
            sr=args.sr
        )
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
