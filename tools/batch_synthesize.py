"""
批量调用阿里云语音合成API (基于123.py的正确方式)
一次连接，发送多条文本
"""

import os
import base64
import threading
import time
import wave
import random
import json

import dashscope
from dashscope.audio.qwen_tts_realtime import *

# ============== 配置 ==============
API_KEY = 'sk-322570c89a1a42c68147cfabea6a8c3e'
OUTPUT_DIR = 'voice_dataset'
# VOICE_MODEL = 'qwen-tts-vc-guanyu-voice-20260210145751288-6746'
VOICE_MODEL = 'qwen-tts-vc-guanyu-voice-20260210153303233-7f8f'
# ============== 300条随机文本 ==============
TEXTS = [
    # 原始6条
    '对吧~我就特别喜欢这种超市，',
    '尤其是过年的时候',
    '去逛超市',
    '就会觉得',
    '超级超级开心！',
    '想买好多好多的东西呢！',
    
    # === 问候语 (30条) ===
    '你好啊，今天天气真不错！',
    '很高兴见到你！',
    '早上好，新的一天开始了！',
    '好久不见，你还好吗？',
    '嗨，我在这里等你呢！',
    '欢迎回来！',
    '你好，好久不见！',
    '下午好，午安！',
    '晚上好，辛苦了！',
    '晚安，做个好梦！',
    '早上好，今天也要加油哦！',
    '午安，吃饭了吗？',
    '下班了，路上小心！',
    '周末愉快！',
    '节日快乐！',
    '新年快乐，恭喜发财！',
    '生日粗卡哈密达！',
    '一路顺风！',
    '旅途愉快！',
    '保重身体哦！',
    '注意安全！',
    '想你啦！',
    '爱你哟！',
    '么么哒！',
    '晚安咯！',
    '早安呀！',
    '嗨起来！',
    '冲鸭！',
    '奥利给！',
    
    # === 情感表达 (30条) ===
    '我今天特别开心！',
    '真是太棒了！',
    '太感动了，眼泪都要流下来了！',
    '有点小失落呢...',
    '超级无敌开心！',
    '哎呀，好尴尬啊！',
    '吓我一跳！',
    '太惊喜了！',
    '有点紧张呢...',
    '我现在很平静。',
    '生气气！',
    '难过死了...',
    '兴奋得睡不着！',
    '后悔死了！',
    '太紧张了！',
    '松了一口气！',
    '郁闷啊！',
    '爽歪歪！',
    '哭唧唧...',
    '笑死我了！',
    '无语了...',
    '崩溃了！',
    '美滋滋！',
    '气鼓鼓！',
    '暖洋洋！',
    '酸了酸了！',
    '离谱！',
    '絕了！',
    '裂开了！',
    '芜湖起飞！',
    
    # === 日常对话 (50条) ===
    '这个多少钱？',
    '给我来一杯奶茶！',
    '我想买这个，太好看了！',
    '这个有优惠吗？',
    '老板，便宜点呗！',
    '刷卡还是现金？',
    '帮我包起来，谢谢！',
    '我要退货，不想要了。',
    '这个可以试穿吗？',
    '快递几天能到？',
    '你在哪里？我去找你。',
    '吃什么？随便。',
    '什么时候回来？',
    '为什么？',
    '怎么办？',
    '好看吗？',
    '能听懂吗？',
    '怎么办到的？',
    '远不远？',
    '行不行？',
    '今天堵车堵了一个小时！',
    '家里没电了，看看电表。',
    '衣服洗了，晾一下。',
    '快递到了，去拿一下。',
    '明天要出差，准备东西。',
    '这电视剧好好看！',
    '游戏输了，再来一局！',
    '今天运动了吗？',
    '手机没电了，充电呢。',
    '这歌真好听，单曲循环！',
    '这个方案不行，改一下。',
    '开会了，大家做好准备！',
    '项目进度怎么样了？',
    '这个bug修好了吗？',
    '今天加班，别忘了！',
    '工资涨了，太好了！',
    '我提交了，你看一下。',
    '这个需求很简单嘛。',
    '客户反馈来了，注意看。',
    '下班了，走吧！',
    '明天见！',
    '拜拜！',
    '回头聊！',
    '待会儿见！',
    '一路平安！',
    '保重！',
    '照顾好自己！',
    '别太累了！',
    
    # === 美食相关 (30条) ===
    '我要吃火锅，麻辣的那种！',
    '这顿饭我请客！',
    '好饿啊，想吃烤肉！',
    '来一碗牛肉面，多加葱花！',
    '这个蛋糕看起来好好吃！',
    '给我来杯咖啡，不加糖。',
    '外卖到了，下来拿一下。',
    '今天吃素，清淡一点。',
    '这个菜有点咸了。',
    '渴了，来杯冰可乐！',
    '想喝奶茶，珍珠奶茶！',
    '这个太辣了，受不了！',
    '给我来碗米饭！',
    '有没有清淡的？',
    '这个口味不错！',
    '太油腻了！',
    '来碗汤暖暖胃！',
    '甜点时间到！',
    '这个水果很甜！',
    '买奶茶吗？买一送一！',
    '烧烤要加辣！',
    '火锅配啤酒！',
    '早餐吃什么？',
    '午餐吃食堂吧。',
    '晚餐自己做！',
    '夜宵走起！',
    '减脂餐安排上！',
    '今天吃顿好的！',
    '这个味道绝了！',
    '光盘行动！',
    
    # === 数字念读 (30条) ===
    '一二三四五六七八九十！',
    '一百二十三块五毛！',
    '电话号码是一三八六八八八八八八八！',
    '今天星期几？星期三！',
    '零一二三，一二三四！',
    '一二三四五，上山打老虎！',
    '芝麻开门！',
    '一加一等于二！',
    '三二一，开始！',
    '第一第二第三第四第五！',
    '十二十三十四十五！',
    '一百两百三百四百五百！',
    '一九九八二零二三！',
    '零到九，一二三四五六七八九！',
    '个十百千万！',
    '一二三四五六七八九十一十二！',
    '一二三，预备，唱！',
    '五十六个民族，五十六枝花！',
    '一二三四五六七八九十个！',
    '一二三四五六七八九十百千！',
    '一百一十一！',
    '二百二十二！',
    '三百三十三！',
    '四百四十四！',
    '五百五十五！',
    '六百六十六！',
    '七百七十七！',
    '八百八十八！',
    '九百九十九！',
    '一千一百一十一！',
    
    # === 中文常用句 (100条) ===
    '没问题！',
    '好的好的！',
    '知道了！',
    '明白了！',
    '可以的！',
    '没问题包在我身上！',
    '随便你！',
    '你说呢？',
    '那当然！',
    '必须的！',
    '好像大概也许吧！',
    '我也不知道啊！',
    '别问我，问别人！',
    '你看呢？',
    '你说了算！',
    '都听你的！',
    '可以可以！',
    '行行行！',
    '好好好！',
    '是是是！',
    '对对对！',
    '没错没错！',
    '那必须的！',
    '开玩笑的！',
    '认真的吗？',
    '真的吗？',
    '不至于吧？',
    '太夸张了吧？',
    '有点意思啊！',
    '这不是巧了吗！',
    '缘分呐！',
    '太巧了吧！',
    '世界真小啊！',
    '人生如戏啊！',
    '世事难料啊！',
    '差不多吧！',
    '马马虎虎吧！',
    '还凑合吧！',
    '一般般吧！',
    '不怎么样吧！',
    '也就那样吧！',
    '你忙吧！',
    '不打扰了！',
    '下次再说吧！',
    '以后再聊吧！',
    '有空再聚吧！',
    '下次请你吃饭！',
    '一言为定！',
    '说好了啊！',
    '拉钩上吊！',
    '不许反悔哦！',
    '说话算话！',
    '骗人是小狗！',
    '真的假的？',
    '没骗你！',
    '我发誓！',
    '天地良心！',
    '我对天发誓！',
    '好了好了！',
    '够了够了！',
    '行了行了！',
    '好了，别说了！',
    'stop！',
    '别闹了！',
    '正经点！',
    '认真一点！',
    '严肃点！',
    '淡定淡定！',
    '稍等一下！',
    '等一下！',
    '等会儿！',
    '马上就好！',
    '很快就好！',
    '再给我一分钟！',
    '不要着急！',
    '不要慌！',
    '稳住稳住！',
    '不要怕！',
    '勇敢一点！',
    '加油加油！',
    '你可以的！',
    '相信你哦！',
    '你最棒了！',
    '厉害厉害！',
    '佩服佩服！',
    '牛啊牛啊！',
    '大佬请收下我的膝盖！',
    '打扰了！',
    '告辞！',
    '溜了溜了！',
    '拜拜了您嘞！',
    '下次再战！',
    '后会有期！',
    '山水有相逢！',
    '有缘再见！',
    '珍重！',
]

class MyCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        self.complete_event = threading.Event()
        self.file = open('result_batch.pcm', 'wb')

    def on_open(self):
        print('✓ 连接已建立')

    def on_close(self, close_status_code, close_msg):
        self.file.close()
        print(f'✓ 连接已关闭: {close_status_code}')

    def on_event(self, response):
        try:
            type = response['type']
            if 'session.created' == type:
                print(f'✓ Session创建: {response["session"]["id"]}')
            if 'response.audio.delta' == type:
                recv_audio_b64 = response['delta']
                self.file.write(base64.b64decode(recv_audio_b64))
            if 'response.done' == type:
                print(f'✓ 响应完成')
            if 'session.finished' == type:
                print('✓ 会话完成')
                self.complete_event.set()
        except Exception as e:
            print(f'[Error] {e}')

    def wait_for_finished(self):
        self.complete_event.wait()


def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print('=' * 60)
    print(f'准备合成 {len(TEXTS)} 条文本')
    print('=' * 60)
    
    # 初始化API
    dashscope.api_key = API_KEY
    
    callback = MyCallback()
    
    print('\n连接阿里云TTS...')
    qwen_tts_realtime = QwenTtsRealtime(
        model='qwen3-tts-vc-realtime-2025-11-27',
        callback=callback,
        url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
    )
    
    qwen_tts_realtime.connect()
    qwen_tts_realtime.update_session(
        voice=VOICE_MODEL,
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode='server_commit'
    )
    
    print('\n开始合成文本...')
    for i, text in enumerate(TEXTS, 1):
        print(f'  [{i:02d}/{len(TEXTS)}] {text}')
        qwen_tts_realtime.append_text(text)
        time.sleep(0.1)  # 小延迟
    
    qwen_tts_realtime.finish()
    callback.wait_for_finished()
    
    # 转换为WAV
    pcm_file = 'result_batch.pcm'
    wav_file = os.path.join(OUTPUT_DIR, 'all_voices.wav')
    
    print(f'\n转换 PCM → WAV...')
    try:
        with open(pcm_file, 'rb') as f:
            pcm_data = f.read()
        with wave.open(wav_file, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm_data)
        print(f'✓ 已保存: {wav_file}')
        os.remove(pcm_file)
    except Exception as e:
        print(f'✗ 转换失败: {e}')
    
    # 统计
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
    
    print(f'\n📁 输出目录: {OUTPUT_DIR}')
    print(f'📊 文件数: {len(files)}')
    print(f'💾 总大小: {total_size/1024/1024:.2f} MB')
    print('=' * 60)


if __name__ == '__main__':
    main()
