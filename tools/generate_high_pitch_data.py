"""
生成高音训练数据 - 专门针对高音优化
"""

import os
import wave
import time
import base64
import json
import subprocess

# 阿里云 API 配置
API_KEY = "sk-你的API密钥"

os.makedirs('voice_dataset_high', exist_ok=True)

# 高音文本列表 - 专门选高音、爆发性、夸张的句子
HIGH_PITCH_TEXTS = [
    # 惊叹系列
    '卧槽！',
    '牛逼！',
    '厉害厉害！',
    '我的天！',
    '太牛了吧！',
    '卧槽卧槽！',
    '绝绝子！',
    '逆天！',
    '太夸张了吧！',
    '无敌了！',
    
    # 尖叫系列
    '啊——！',
    '啊啊啊啊啊！',
    '妈呀！',
    '救命啊！',
    '吓死我了！',
    '天呐！',
    '哎呀我去！',
    '我去！',
    '我去我去！',
    '握草！',
    
    # 开心系列
    '哈哈哈哈！',
    '笑死我了！',
    '太搞笑了吧！',
    '嘻嘻嘻！',
    '嘎嘎嘎！',
    '嘿嘿嘿！',
    '乐死我了！',
    '爽！',
    '开心开心！',
    '太快乐了！',
    
    # 伤心系列
    '呜呜呜...',
    '难过...',
    '哭唧唧...',
    '我的心好痛...',
    '扎心了！',
    '哭辽...',
    '悲伤辣么大...',
    '呜...',
    '唉...',
    '算了算了...',
    
    # 着急系列
    '快点快点！',
    '来不及了！',
    '快点啊！',
    '快点快点快点！',
    '催催催！',
    '快快快！',
    '来不及来不及！',
    '快啊快啊！',
    '火烧眉毛了！',
    '紧急紧急！',
    
    # 愤怒系列
    '生气气！',
    '气死了！',
    '过分！',
    '太过分了！',
    '怒了！',
    '哼！',
    '不理你了！',
    '哼╭(╯^╰)╮！',
    '可恶！',
    '靠！',
    
    # 夸张系列
    '超级无敌螺旋爆炸开心！',
    '妈咪妈咪哄！',
    '巴啦啦能量！',
    '代表月亮消灭你！',
    '看我无敌风火轮！',
    '奥利给！干就完了！',
    '、皮一下很开心！',
    '你完蛋了！',
    '走着瞧！',
    '没在怕的！',
    
    # 萌系系列
    '么么哒！',
    '爱你哟！',
    '抱抱！',
    '举高高！',
    '要举高高！',
    '举高高举高高！',
    '贴贴！',
    '蹭蹭！',
    '打打！',
    '摸摸头！',
]

print(f"🎵 准备生成 {len(HIGH_PITCH_TEXTS)} 条高音训练数据...")
print("=" * 60)

def call_alibaba_tts(text, output_file, voice="男声音色"):
    """调用阿里云TTS"""
    cmd = [
        'curl', '-s', '-w', '\n%{http_code}',
        'https://dashscope.aliyuncs.com/api/v1/services/audio/tts/online',
        '-H', f'Authorization: Bearer {API_KEY}',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'model': 'cosyvoice-v1',
            'input': {'text': text},
            'parameters': {
                'voice': voice,
                'format': 'pcm',
                'sample_rate': 16000,
                'volume': 80,
                'rate': 0,
                'pitch': 2,  # 提高音调
            }
        }),
        '-o', output_file
    ]
    subprocess.run(cmd, check=True)

# 快速生成（使用命令行方式）
def generate_tts(text, idx):
    """生成单个音频"""
    output_pcm = f'voice_dataset_high/temp_{idx}.pcm'
    output_wav = f'voice_dataset_high/{idx:03d}_{text}.wav'
    
    # 调用阿里云API
    cmd = [
        'curl', '-s', '-o', output_pcm, '-w', '%{http_code}',
        'https://dashscope.aliyuncs.com/api/v1/services/audio/tts/online',
        '-H', f'Authorization: Bearer {API_KEY}',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({
            'model': 'cosyvoice-v1',
            'input': {'text': text},
            'parameters': {
                'voice': '男声音色',  # 替换为你的声音模型
                'format': 'pcm',
                'sample_rate': 16000,
                'volume': 80,
                'rate': 0,
                'pitch': 3,  # 提高音调
            }
        })
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # 如果失败，用简单的静音代替
        if result.returncode != 0 or not os.path.exists(output_pcm):
            # 创建1秒静音wav
            create_silent_wav(output_wav, duration=0.5)
            return
        
        # 转换为wav
        with wave.open(output_pcm, 'rb') as pcm:
            params = pcm.getparams()
            frames = pcm.readframes(params.nframes)
            with wave.open(output_wav, 'wb') as wav:
                wav.setparams(params)
                wav.writeframes(frames)
        
        os.remove(output_pcm)
    except Exception as e:
        create_silent_wav(output_wav, duration=0.5)

def create_silent_wav(filepath, duration=1.0):
    """创建静音wav"""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    with wave.open(filepath, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b'\x00' * num_samples * 2)

# 生成所有文件
for i, text in enumerate(HIGH_PITCH_TEXTS, 1):
    output_wav = f'voice_dataset_high/{i:03d}_{text}.wav'
    
    # 创建静音wav作为占位符（实际使用时需要阿里云API）
    create_silent_wav(output_wav, duration=1.0 + len(text) * 0.1)
    
    if i % 10 == 0:
        print(f'  进度: {i}/{len(HIGH_PITCH_TEXTS)}')

print("=" * 60)
print(f"✅ 生成完成！共 {len(HIGH_PITCH_TEXTS)} 个文件")
print(f"📁 保存位置: voice_dataset_high/")
print()
print("⚠️  注意：上面的代码使用了静音占位符")
print("   实际使用时，需要配置正确的阿里云 API Key")
print("   并使用阿里云 SDK 或正确的 API 调用方式生成真实音频")
print()
print("=" * 60)
print("💡 建议：")
print("   1. 去阿里云控制台生成这些高音文本的音频")
print("   2. 或者手动录制一些高音样本")
print("   3. 将生成的文件添加到 voice_dataset/wavs/ 目录")
print("   4. 重新训练模型")
