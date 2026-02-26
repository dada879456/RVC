"""
从现有训练数据生成高音变体 - 用 pitch shift
"""

import os
import librosa
import numpy as np
import wave

INPUT_WAV = 'voice_dataset/all_voices.wav'
OUTPUT_DIR = 'voice_dataset/high_pitch'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_wav(filepath, y, sr=16000):
    """保存为16bit WAV"""
    y_int16 = (y * 32767).astype('int16')
    with wave.open(filepath, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y_int16.tobytes())

print(f"📂 加载音频: {INPUT_WAV}")
y, sr = librosa.load(INPUT_WAV, sr=16000)
print(f"⏱️  时长: {len(y)/sr:.1f}秒")

print("\n🎵 生成音高偏移版本...")

# 生成不同音高偏移版本
pitch_shifts = [
    (+3, "+3半音(高八度)"),
    (+4, "+4半音"),
    (+5, "+5半音"),
    (+6, "+6半音"),
    (+8, "+8半音(很高)"),
    (+10, "+10半音(极限)"),
]

created = []
for semitones, desc in pitch_shifts:
    print(f"  生成: {desc}")
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    
    filename = f"pitch_{semitones:+d}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    save_wav(filepath, y_shifted)
    created.append((filename, len(y_shifted)/sr))

print("\n" + "=" * 50)
print(f"✅ 已生成 {len(created)} 个高音版本")
print(f"📁 保存位置: {OUTPUT_DIR}/")
print()

# 合并所有高音文件到一个目录
print("📦 合并高音文件到 wavs 目录...")
wavs_dir = 'voice_dataset/wavs'
for semitones, desc in pitch_shifts:
    src = os.path.join(OUTPUT_DIR, f"pitch_{semitones:+d}.wav")
    dst = os.path.join(wavs_dir, f"high_pitch_{semitones:+d}.wav")
    import shutil
    shutil.copy2(src, dst)
    print(f"  ✓ {os.path.basename(dst)}")

print("\n" + "=" * 50)
print("📊 最终训练数据统计:")
wavs_count = len([f for f in os.listdir(wavs_dir) if f.endswith('.wav')])
print(f"   总文件数: {wavs_count} 个")

# 计算总时长
total_duration = 0
for f in os.listdir(wavs_dir):
    if f.endswith('.wav'):
        y_temp, _ = librosa.load(os.path.join(wavs_dir, f), sr=16000)
        total_duration += len(y_temp) / 16000

print(f"   总时长: {total_duration/60:.1f} 分钟")

print("\n💡 现在你可以用增强后的数据重新训练了！")
print("  高音版本会帮助模型学习高音特征，减少电音问题。")
