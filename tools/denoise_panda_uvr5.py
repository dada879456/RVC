#!/usr/bin/env python3
"""
熊猫叫声音频处理工具
使用 UVR5 模型进行人声/叫声提取和去回声
保留叫声，去除背景噪音
"""

import os
import sys
import argparse
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import librosa

# 添加项目路径到 Python 路径
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "infer"))
sys.path.append(os.path.join(now_dir, "configs"))

# 设置环境变量
weight_uvr5_root = os.path.join(now_dir, "assets", "uvr5_weights")
os.environ["weight_uvr5_root"] = weight_uvr5_root


def run_ffmpeg(cmd):
    """运行 ffmpeg 命令"""
    subprocess.run(cmd, check=True, stdout=subprocess.DEVICE, stderr=subprocess.DEVICE)


def convert_audio(input_path, output_path, sr=44100):
    """
    使用 ffmpeg 转换音频格式
    """
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ac', '2', '-ar', str(sr),
        output_path, '-y'
    ]
    run_ffmpeg(cmd)


def get_audio_info(input_path):
    """获取音频信息"""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=channels,sample_rate', '-of', 'csv=p=0', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        parts = result.stdout.strip().split(',')
        return {'channels': int(parts[0]), 'sample_rate': int(parts[1])}
    return None


def uvr5_dereverb(input_path, output_path, model_name="VR-DeEchoNormal"):
    """
    使用 UVR5 模型进行去回声/去混响
    """
    from infer.modules.uvr5.mdxnet import MDXNetDereverb
    from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_half = True

    print(f"  使用模型: {model_name}")
    print(f"  设备: {device}")

    # 临时文件路径
    temp_wav = input_path

    # 检查是否需要格式转换
    info = get_audio_info(input_path)
    if info is None or info['channels'] != 2 or info['sample_rate'] != 44100:
        print(f"  格式转换...")
        temp_wav = os.path.join(tempfile.gettempdir(), "temp_uvr5.wav")
        convert_audio(input_path, temp_wav)

    # 选择模型
    if model_name == "onnx_dereverb_By_FoxJoy":
        pre_fun = MDXNetDereverb(15, device)
        pre_fun.pred.prediction(
            temp_wav,
            output_path,
            output_path.replace(".wav", "_others.wav"),
            "wav"
        )
    else:
        model_path = os.path.join(weight_uvr5_root, model_name + ".pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        if "DeEcho" in model_name:
            func = AudioPreDeEcho
        else:
            func = AudioPre

        pre_fun = func(
            agg=7,
            model_path=model_path,
            device=device,
            is_half=is_half
        )

        # UVR5 输出人声(vocal)和伴奏(others)
        # 对于熊猫叫声，我们要的是"人声"部分
        pre_fun._path_audio_(
            temp_wav,
            output_path.replace(".wav", "_others.wav"),
            output_path,
            "wav"
        )

    # 清理临时文件
    if temp_wav != input_path and os.path.exists(temp_wav):
        os.remove(temp_wav)

    print(f"  UVR5 处理完成")


def simple_dereverb(audio, sr):
    """
    简单的去混响算法
    """
    delay_samples = int(0.02 * sr)
    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples]
    reverb_estimate = delayed * 0.3
    clean_audio = audio - reverb_estimate
    max_val = np.max(np.abs(clean_audio))
    if max_val > 0:
        clean_audio = clean_audio / max_val * 0.95
    return clean_audio


def remove_noise_simple(audio, sr):
    """
    简单降噪
    """
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    noise_threshold = np.percentile(magnitude, 10)
    noise_mask = magnitude < noise_threshold * 3
    magnitude_clean = np.where(noise_mask, magnitude * 0.7, magnitude)

    stft_clean = magnitude_clean * np.exp(1j * np.angle(stft))
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length)

    return audio_clean


def process_panda_audio(input_path, output_path, use_uvr5=True, model_name="VR-DeEchoNormal"):
    """
    处理熊猫叫声
    """
    print(f"正在加载音频: {input_path}")
    audio, orig_sr = librosa.load(input_path, sr=44100)
    print(f"音频时长: {len(audio)/44100:.2f}秒")

    if use_uvr5:
        print("使用 UVR5 模型进行去回声...")
        try:
            temp_input = os.path.join(tempfile.gettempdir(), "temp_panda_input.wav")
            sf.write(temp_input, audio, 44100)

            uvr5_dereverb(temp_input, output_path, model_name)

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                print("  UVR5 处理失败，使用简单方法...")
                raise Exception("UVR5 failed")
        except Exception as e:
            print(f"  UVR5 错误: {e}")
            print("  使用简单去混响...")
            audio_clean = simple_dereverb(audio, 44100)
            audio_clean = remove_noise_simple(audio_clean, 44100)
            sf.write(output_path, audio_clean, 44100)
    else:
        print("使用简单去混响...")
        audio_clean = simple_dereverb(audio, 44100)
        audio_clean = remove_noise_simple(audio_clean, 44100)
        sf.write(output_path, audio_clean, 44100)

    result_audio, _ = librosa.load(output_path, sr=44100)
    print(f"处理完成！")
    print(f"输出: {output_path}")
    print(f"处理后时长: {len(result_audio)/44100:.2f}秒")


def batch_process(input_dir, output_dir, use_uvr5=True, model_name="VR-DeEchoNormal"):
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
        output_path = os.path.join(output_dir, f"clean_{os.path.splitext(file)[0]}.wav")
        print(f"\n[{i}/{len(files)}] 正在处理: {file}")
        process_panda_audio(input_path, output_path, use_uvr5, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='熊猫叫声音频处理工具 - 使用 UVR5 去回声')
    parser.add_argument('input', help='输入音频文件或目录')
    parser.add_argument('-o', '--output', default=None,
                       help='输出文件或目录')
    parser.add_argument('--no-uvr5', action='store_true',
                       help='不使用 UVR5，使用简单去混响')
    parser.add_argument('-m', '--model', default='VR-DeEchoNormal',
                       choices=['VR-DeEchoNormal', 'VR-DeEchoDeReverb', 'VR-DeEchoAggressive', 'onnx_dereverb_By_FoxJoy'],
                       help='UVR5 模型')
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
            use_uvr5=not args.no_uvr5,
            model_name=args.model
        )
    elif os.path.isdir(args.input):
        if args.output is None:
            output_dir = os.path.join(args.input, 'cleaned')
        else:
            output_dir = args.output

        batch_process(
            args.input, output_dir,
            use_uvr5=not args.no_uvr5,
            model_name=args.model
        )
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
