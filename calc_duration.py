import os
import struct

total_duration = 0
audio_dir = 'training_audio'

print('=' * 50)
print('音频时长计算')
print('=' * 50)

for filename in sorted(os.listdir(audio_dir)):
    if filename.endswith('.wav'):
        filepath = os.path.join(audio_dir, filename)
        
        with open(filepath, 'rb') as f:
            # 读取 WAV 文件头
            f.read(4)  # RIFF
            f.read(4)  # File size
            f.read(4)  # WAVE
            f.read(4)  # fmt
            chunk_size = struct.unpack('<I', f.read(4))[0]
            audio_format = struct.unpack('<H', f.read(2))[0]
            num_channels = struct.unpack('<H', f.read(2))[0]
            sample_rate = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # byte_rate
            f.read(2)  # block_align
            bits_per_sample = struct.unpack('<H', f.read(2))[0]
            
            file_size = os.path.getsize(filepath)
            data_size = file_size - 44
            byte_rate = sample_rate * num_channels * bits_per_sample // 8
            duration = data_size / byte_rate
            
            total_duration += duration
            print(f'{filename}: {duration:.2f}秒')

print('=' * 50)
print(f'总时长: {total_duration:.2f} 秒')
print(f'约 {total_duration/60:.2f} 分钟')
print(f'约 {total_duration/3600:.2f} 小时')
print('=' * 50)
