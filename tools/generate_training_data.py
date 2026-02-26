"""
RVC 训练数据增强脚本
将阿里云克隆的声音进行多种增强，扩充训练数据
"""

import os
import numpy as np
import librosa
import soundfile as sf

def load_audio(path, sr=16000):
    """加载音频"""
    y, _ = librosa.load(path, sr=sr)
    return y

def save_audio(y, path, sr=16000):
    """保存音频"""
    sf.write(path, y, sr)
    print(f"✓ 保存: {path}")

def pitch_shift(y, sr, n_steps):
    """变调"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    """时间拉伸"""
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise(y, noise_level=0.005):
    """添加微量噪声"""
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def change_volume(y, level=1.0):
    """调整音量"""
    return y * level

def main():
    source_file = "result_24k.wav"
    output_dir = "training_data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载原始音频
    print(f"加载: {source_file}")
    y = load_audio(source_file)
    print(f"原始时长: {len(y)/16000:.2f}秒")
    
    # 原始文件
    save_audio(y, os.path.join(output_dir, "origin_01.wav"))
    save_audio(y, os.path.join(output_dir, "origin_02.wav"))
    save_audio(y, os.path.join(output_dir, "origin_03.wav"))
    
    # 变调版本 (-4 到 +4)
    for semitones in [-4, -2, -1, 1, 2, 4]:
        y_pitch = pitch_shift(y, 16000, semitones)
        save_audio(y_pitch, os.path.join(output_dir, f"pitch_{semitones:+d}.wav"))
    
    # 时间拉伸 (0.9x, 1.1x)
    for rate in [0.9, 1.1]:
        y_stretch = time_stretch(y, rate)
        rate_str = str(rate).replace('.', '')
        save_audio(y_stretch, os.path.join(output_dir, f"stretch_{rate_str}.wav"))
    
    # 音量调整
    for vol in [0.7, 1.3]:
        y_vol = change_volume(y, vol)
        save_audio(y_vol, os.path.join(output_dir, f"vol_{int(vol*10)}.wav"))
    
    # 组合增强
    y_pitch_low = pitch_shift(y, 16000, -2)
    y_pitch_low_vol = change_volume(y_pitch_low, 0.8)
    save_audio(y_pitch_low_vol, os.path.join(output_dir, "combo_low.wav"))
    
    y_pitch_high = pitch_shift(y, 16000, 2)
    y_pitch_high_vol = change_volume(y_pitch_high, 1.2)
    save_audio(y_pitch_high_vol, os.path.join(output_dir, "combo_high.wav"))
    
    # 统计
    files = os.listdir(output_dir)
    total_duration = len(files) * len(y) / 16000
    
    print("\n" + "=" * 50)
    print("生成完成!")
    print(f"文件数: {len(files)}")
    print(f"总时长: {total_duration:.1f}秒 (约 {total_duration/60:.1f}分钟)")
    print(f"输出目录: {output_dir}")
    print("=" * 50)
    
    return output_dir

if __name__ == "__main__":
    main()
