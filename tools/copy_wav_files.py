"""
复制 menglan1.wav 为 menglan9.wav 到 menglan50.wav
"""

import os
import shutil

source_file = "dada/menglan1.wav"
output_dir = "dada"

for i in range(9, 51):
    output_file = os.path.join(output_dir, f"menglan{i}.wav")
    shutil.copy2(source_file, output_file)
    print(f"✓ 已复制: menglan{i}.wav")

print(f"\n完成！共复制了 42 个文件 (menglan9.wav 到 menglan50.wav)")
