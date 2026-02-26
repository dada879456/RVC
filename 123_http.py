import os
import dashscope
import base64
import urllib.request

# 设置 API Key
if 'DASHSCOPE_API_KEY' in os.environ:
    dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
else:
    dashscope.api_key = 'sk-322570c89a1a42c68147cfabea6a8c3e'

text_to_synthesize = [
    '对吧~我就特别喜欢这种超市，',
    '尤其是过年的时候',
    '去逛超市',
    '就会觉得',
    '超级超级开心！',
    '想买好多好多的东西呢！'
]

def synthesize_tts(text, voice="Bella"):
    """使用 HTTP API 进行 TTS 合成"""
    response = dashscope.MultiModalConversation.call(
        model="qwen3-tts-flash",
        text=text,
        voice=voice,
        language_type="Chinese",
        stream=False
    )

    if response.status_code == 200:
        # 提取音频 URL
        audio_obj = response.output.audio
        audio_url = audio_obj.url
        print(f"音频URL: {audio_url}")

        # 下载音频文件
        with urllib.request.urlopen(audio_url) as response:
            audio_data = response.read()
        print(f"下载完成: {len(audio_data)} bytes")
        return audio_data
    else:
        print(f"错误: {response.message}")
        return None

if __name__ == '__main__':
    print('=== 开始合成音频 ===')

    # 合并所有文本
    full_text = ''.join(text_to_synthesize)
    print('合成文本: ' + full_text)

    # 合成音频
    audio_data = synthesize_tts(full_text, voice="Bella")

    if audio_data:
        # 保存为 WAV 文件
        with open('result_http.wav', 'wb') as f:
            f.write(audio_data)
        print(f'[OK] 已保存音频: result_http.wav ({len(audio_data)} bytes)')
    else:
        print('[ERROR] 音频合成失败')
