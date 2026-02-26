import os
import time
import dashscope
from dashscope.audio.tts_v2 import VoiceEnrollmentService, SpeechSynthesizer

# =============================================
# 配置区域
# =============================================

# API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-322570c89a1a42c68147cfabea6a8c3e"
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

# 复刻参数
TARGET_MODEL = "cosyvoice-v3-plus"
VOICE_PREFIX = "trainvoice"  # 仅允许数字和小写字母，小于十个字符

# 参考音频 URL (公网可访问)
# 说明: 需要是人声朗读的音频，不要静音或纯音乐
# 可以使用 menglan.MP3 (约3MB，清晰人声)
AUDIO_URL = "https://998555.oss-cn-beijing.aliyuncs.com/menglan.MP3"

# 训练文本列表 (可以扩充)
TRAINING_TEXTS = [
    "你好，欢迎使用语音合成系统。",
    "今天天气真好，适合出门散步。",
    "人工智能技术发展迅速。",
    "音乐是人类共同的语言。",
    "学习新技术需要不断练习。",
    "编程让世界变得更美好。",
    "保持积极乐观的心态很重要。",
    "成功的关键在于坚持不懈。",
    "团结协作能够创造奇迹。",
    "健康是人生最大的财富。",
    "知识改变命运，学习成就未来。",
    "每个人都有自己的梦想和追求。",
    "珍惜当下的每一刻时光。",
    "勇敢面对挑战，永不放弃。",
    "微笑是最好的沟通方式。",
    "勤奋是成功的朋友。",
    "时间是最宝贵的资源。",
    "相信自己，你一定能行。",
    "失败是成功之母。",
    "生活就像一面镜子，你对它笑，它就对你笑。",
    "持续的进步比一次的成功更重要。",
    "阅读可以开阔视野，增长见识。",
    "良好的沟通能够化解矛盾。",
    "创新思维是发展的动力。",
    "自律给你自由。",
    "感恩的心，感谢有你。",
    "坚持就是胜利。",
    "生命在于运动，健康在于饮食。",
    "梦想还是要有的，万一实现了呢。",
    "活到老，学到老。",
]

# 输出配置
OUTPUT_DIR = "training_data_generated"
SAMPLE_RATE = 40000  # RVC 推荐采样率

# =============================================
# 步骤 1: 复刻音色
# =============================================
print("=" * 50)
print("Step 1: 开始复刻音色")
print("=" * 50)

service = VoiceEnrollmentService()
try:
    voice_id = service.create_voice(
        target_model=TARGET_MODEL,
        prefix=VOICE_PREFIX,
        url=AUDIO_URL
    )
    print(f"[OK] 音色复刻请求已提交")
    print(f"Request ID: {service.get_last_request_id()}")
    print(f"Voice ID: {voice_id}")
except Exception as e:
    print(f"[ERROR] 音色复刻失败: {e}")
    raise e

# =============================================
# 步骤 2: 轮询查询音色状态
# =============================================
print("\n" + "=" * 50)
print("Step 2: 等待音色复刻完成")
print("=" * 50)

max_attempts = 60  # 最多等待 10 分钟
poll_interval = 10  # 每 10 秒查询一次

for attempt in range(max_attempts):
    try:
        voice_info = service.query_voice(voice_id=voice_id)
        status = voice_info.get("status")
        print(f"尝试 {attempt + 1}/{max_attempts}: 状态 = '{status}'")

        if status == "OK":
            print("\n[OK] 音色复刻成功！")
            print(f"音色信息: {voice_info}")
            break
        elif status == "UNDEPLOYED":
            print(f"[ERROR] 音色处理失败: {status}")
            raise RuntimeError(f"音色处理失败: {status}")
        elif status in ["PROCESSING", "DEPLOYING"]:
            print("  等待中...")
        else:
            print(f"  未知状态: {status}")

        time.sleep(poll_interval)
    except Exception as e:
        print(f"[ERROR] 查询失败: {e}")
        time.sleep(poll_interval)
else:
    print("[ERROR] 等待超时，音色尚未准备完成")
    raise RuntimeError("等待超时")

# =============================================
# 步骤 3: 创建输出目录
# =============================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n[OK] 输出目录已创建: {OUTPUT_DIR}")

# =============================================
# 步骤 4: 批量生成训练音频
# =============================================
print("\n" + "=" * 50)
print("Step 3: 开始批量生成训练音频")
print("=" * 50)

try:
    synthesizer = SpeechSynthesizer(model=TARGET_MODEL, voice=voice_id)

    successful_count = 0
    failed_count = 0

    for i, text in enumerate(TRAINING_TEXTS):
        # 生成文件名 (4位数字前缀)
        output_file = os.path.join(OUTPUT_DIR, f"{i+1:04d}.wav")
        text_file = os.path.join(OUTPUT_DIR, f"{i+1:04d}.txt")

        print(f"\n[{i+1}/{len(TRAINING_TEXTS)}] 生成中...")
        print(f"  文本: {text}")

        try:
            # 生成音频
            audio_data = synthesizer.call(text)
            print(f"  [OK] 音频生成成功, {len(audio_data)} bytes")

            # 保存音频文件
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"  [OK] 已保存: {output_file}")

            # 保存对应的文本文件
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  [OK] 已保存: {text_file}")

            successful_count += 1

        except Exception as e:
            print(f"  [ERROR] 生成失败: {e}")
            failed_count += 1

        # 避免请求过快，添加短暂延迟
        time.sleep(0.5)

    # =============================================
    # 完成
    # =============================================
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)
    print(f"成功生成: {successful_count} 个音频")
    print(f"失败数量: {failed_count} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n生成的音频可用于 RVC 模型训练")

except Exception as e:
    print(f"[ERROR] 合成过程出错: {e}")
    raise e
