#api for 240604 release version by Xiaokai
import os
import sys
import json
import re
import time
import librosa
import torch
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat
import sounddevice as sd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
import uvicorn
import logging
import requests
import alibabacloud_oss_v2 as oss
from datetime import datetime
from multiprocessing import Queue, Process, cpu_count, freeze_support
import random
import pymysql
from contextlib import contextmanager
import uuid

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================
# 数据库配置
# =============================================
DB_CONFIG = {
    "host": os.environ.get("DATASET_DB_HOST", "120.24.146.254"),
    "port": int(os.environ.get("DATASET_DB_PORT", 3306)),
    "user": os.environ.get("DATASET_DB_USER", "root"),
    "password": os.environ.get("DATASET_DB_PASSWORD", "842523563"),
    "database": os.environ.get("DATASET_DB_NAME", "dify_mars_dev"),
    "charset": "utf8mb4"
}

# 训练任务状态
TASK_STATUS = {
    "PENDING": "pending",        # 等待处理
    "CLONING": "cloning",       # 音色克隆中
    "GENERATING": "generating",  # 音频生成中
    "COMPLETED": "completed",   # 完成
    "FAILED": "failed"           # 失败
}

@contextmanager
def get_db_connection():
    """获取数据库连接"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def init_training_task_table():
    """初始化训练任务表"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS rvc_training_tasks (
        uid VARCHAR(64) PRIMARY KEY COMMENT '任务唯一ID',
        user_id VARCHAR(64) COMMENT '用户ID',
        audio_url TEXT COMMENT '参考音频URL',
        voice_prefix VARCHAR(32) COMMENT '音色前缀',
        target_duration_min INT DEFAULT 15 COMMENT '目标时长(分钟)',
        output_dir VARCHAR(256) COMMENT '输出目录',
        voice_id VARCHAR(128) COMMENT '阿里云音色ID',
        status VARCHAR(32) DEFAULT 'pending' COMMENT '任务状态',
        total_texts INT DEFAULT 0 COMMENT '总文本数',
        successful_count INT DEFAULT 0 COMMENT '成功数量',
        failed_count INT DEFAULT 0 COMMENT '失败数量',
        estimated_duration_min FLOAT DEFAULT 0 COMMENT '预估时长(分钟)',
        error_message TEXT COMMENT '错误信息',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
        completed_at DATETIME NULL COMMENT '完成时间',
        INDEX idx_user_id (user_id),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RVC训练任务表';
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
            conn.commit()
        logger.info("训练任务表初始化完成")
    except Exception as e:
        logger.warning(f"初始化任务表失败: {e}")


def create_training_task(uid: str, user_id: str, audio_url: str, voice_prefix: str,
                         target_duration_min: int, output_dir: str) -> dict:
    """创建训练任务记录"""
    sql = """
    INSERT INTO rvc_training_tasks
    (uid, user_id, audio_url, voice_prefix, target_duration_min, output_dir, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (uid, user_id, audio_url, voice_prefix,
                                target_duration_min, output_dir, TASK_STATUS["PENDING"]))
        conn.commit()

    return {
        "uid": uid,
        "status": TASK_STATUS["PENDING"]
    }


def update_task_status(uid: str, status: str, **kwargs):
    """更新任务状态"""
    allowed_fields = ["voice_id", "total_texts", "successful_count",
                     "failed_count", "estimated_duration_min", "error_message"]

    set_clause = ["status = %s", "updated_at = NOW()"]
    values = [status]

    for field, value in kwargs.items():
        if field in allowed_fields and value is not None:
            set_clause.append(f"{field} = %s")
            values.append(value)

    if status == TASK_STATUS["COMPLETED"] or status == TASK_STATUS["FAILED"]:
        set_clause.append("completed_at = NOW()")

    values.append(uid)
    sql = f"UPDATE rvc_training_tasks SET {', '.join(set_clause)} WHERE uid = %s"

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, values)
        conn.commit()


def get_task_status(uid: str) -> dict:
    """获取任务状态"""
    sql = """
    SELECT uid, user_id, audio_url, voice_prefix, target_duration_min, output_dir,
           voice_id, status, total_texts, successful_count, failed_count,
           estimated_duration_min, error_message, created_at, updated_at, completed_at
    FROM rvc_training_tasks
    WHERE uid = %s
    """
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, (uid,))
            result = cursor.fetchone()

    if result:
        # 转换日期时间为字符串
        for key in ['created_at', 'updated_at', 'completed_at']:
            if result.get(key):
                result[key] = result[key].strftime('%Y-%m-%d %H:%M:%S')

    return result


def get_user_training_tasks(user_id: str, limit: int = 10) -> list:
    """获取用户的所有训练数据生成任务"""
    sql = """
    SELECT uid, user_id, audio_url, voice_prefix, target_duration_min, output_dir,
           voice_id, status, total_texts, successful_count, failed_count,
           estimated_duration_min, error_message, created_at, updated_at, completed_at
    FROM rvc_training_tasks
    WHERE user_id = %s
    ORDER BY created_at DESC
    LIMIT %s
    """
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, (user_id, limit))
            results = cursor.fetchall()

    for result in results:
        for key in ['created_at', 'updated_at', 'completed_at']:
            if result.get(key):
                result[key] = result[key].strftime('%Y-%m-%d %H:%M:%S')

    return results


def get_user_model_train_tasks(user_id: str, limit: int = 10) -> list:
    """获取用户的所有模型训练任务"""
    sql = """
    SELECT uid, user_id, model_name, data_dir, sample_rate, version, epochs, batch_size, gpu,
           status, current_epoch, total_epochs, loss_g, loss_d, model_path, error_message,
           created_at, updated_at, completed_at
    FROM rvc_model_train_tasks
    WHERE user_id = %s
    ORDER BY created_at DESC
    LIMIT %s
    """
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, (user_id, limit))
            results = cursor.fetchall()

    for result in results:
        for key in ['created_at', 'updated_at', 'completed_at']:
            if result.get(key):
                result[key] = result[key].strftime('%Y-%m-%d %H:%M:%S')

    return results


# 启动时初始化数据库表
init_training_task_table()


# =============================================
# 模型训练任务数据库操作
# =============================================

# 模型训练任务状态
MODEL_TRAIN_STATUS = {
    "PENDING": "pending",        # 等待处理
    "PREPROCESSING": "preprocessing",  # 预处理中
    "EXTRACTING": "extracting",   # 特征提取中
    "TRAINING": "training",       # 训练中
    "COMPLETED": "completed",     # 完成
    "FAILED": "failed"            # 失败
}

def init_model_train_table():
    """初始化模型训练任务表"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS rvc_model_train_tasks (
        uid VARCHAR(64) PRIMARY KEY COMMENT '任务唯一ID',
        user_id VARCHAR(64) COMMENT '用户ID',
        model_name VARCHAR(64) COMMENT '模型名称',
        data_dir VARCHAR(256) COMMENT '训练数据目录',
        sample_rate INT DEFAULT 48000 COMMENT '采样率',
        version VARCHAR(8) DEFAULT 'v2' COMMENT '版本',
        epochs INT DEFAULT 100 COMMENT '训练轮数',
        batch_size INT DEFAULT 4 COMMENT '批次大小',
        gpu VARCHAR(16) DEFAULT '0' COMMENT 'GPU编号',
        status VARCHAR(32) DEFAULT 'pending' COMMENT '任务状态',
        current_epoch INT DEFAULT 0 COMMENT '当前轮数',
        total_epochs INT DEFAULT 0 COMMENT '总轮数',
        loss_g FLOAT DEFAULT 0 COMMENT '生成器损失',
        loss_d FLOAT DEFAULT 0 COMMENT '判别器损失',
        model_path VARCHAR(256) COMMENT '模型文件路径',
        error_message TEXT COMMENT '错误信息',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
        completed_at DATETIME NULL COMMENT '完成时间',
        INDEX idx_user_id (user_id),
        INDEX idx_model_name (model_name),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RVC模型训练任务表';
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
            conn.commit()
        logger.info("模型训练任务表初始化完成")
    except Exception as e:
        logger.warning(f"初始化模型训练任务表失败: {e}")


def create_model_train_task(uid: str, user_id: str, model_name: str, data_dir: str,
                             sample_rate: int, version: str, epochs: int,
                             batch_size: int, gpu: str) -> dict:
    """创建模型训练任务记录"""
    sql = """
    INSERT INTO rvc_model_train_tasks
    (uid, user_id, model_name, data_dir, sample_rate, version, epochs, batch_size, gpu, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (uid, user_id, model_name, data_dir, sample_rate,
                                version, epochs, batch_size, gpu, MODEL_TRAIN_STATUS["PENDING"]))
        conn.commit()

    return {
        "uid": uid,
        "status": MODEL_TRAIN_STATUS["PENDING"]
    }


def update_model_train_status(uid: str, status: str, **kwargs):
    """更新模型训练任务状态"""
    allowed_fields = ["current_epoch", "total_epochs", "loss_g", "loss_d", "model_path", "error_message"]

    set_clause = ["status = %s", "updated_at = NOW()"]
    values = [status]

    for field, value in kwargs.items():
        if field in allowed_fields and value is not None:
            set_clause.append(f"{field} = %s")
            values.append(value)

    if status == MODEL_TRAIN_STATUS["COMPLETED"] or status == MODEL_TRAIN_STATUS["FAILED"]:
        set_clause.append("completed_at = NOW()")

    values.append(uid)
    sql = f"UPDATE rvc_model_train_tasks SET {', '.join(set_clause)} WHERE uid = %s"

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, values)
        conn.commit()


def get_model_train_status(uid: str) -> dict:
    """获取模型训练任务状态"""
    sql = """
    SELECT uid, user_id, model_name, data_dir, sample_rate, version, epochs, batch_size, gpu,
           status, current_epoch, total_epochs, loss_g, loss_d, model_path, error_message,
           created_at, updated_at, completed_at
    FROM rvc_model_train_tasks
    WHERE uid = %s
    """
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql, (uid,))
            result = cursor.fetchone()

    if result:
        # 转换日期时间为字符串
        for key in ['created_at', 'updated_at', 'completed_at']:
            if result.get(key):
                result[key] = result[key].strftime('%Y-%m-%d %H:%M:%S')

    return result


# 初始化模型训练任务表
init_model_train_table()


# =============================================
# 扩充的训练文本库 (用于生成15-30分钟的训练数据)
# =============================================
TRAINING_TEXTS_POOL = [
    # 简短句子 (2-4秒)
    "你好，欢迎使用。",
    "今天天气真好。",
    "早上好。",
    "晚安。",
    "再见。",
    "谢谢。",
    "对不起。",
    "没关系。",
    "请问您找谁？",
    "请稍等。",
    "好的，我明白了。",
    "好的，没问题。",
    "我来帮您。",
    "您稍等一下。",
    "请这边走。",
    "请坐。",
    "喝水吗？",
    "吃了吗？",
    "去哪里？",
    "什么时候？",
    "多少钱？",
    "太贵了。",
    "便宜一点吧。",
    "我考虑一下。",
    "就要这个。",
    "中等大小。",
    "要热的。",
    "不要辣。",
    "少放盐。",
    "打包带走。",

    # 中等句子 (4-6秒)
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
    "生活就像一面镜子。",
    "持续的进步比成功更重要。",
    "阅读可以开阔视野。",
    "良好的沟通化解矛盾。",
    "创新思维是发展动力。",
    "自律给你自由。",
    "感恩的心，感谢有你。",
    "坚持就是胜利。",
    "生命在于运动。",
    "梦想还是要有的。",
    "活到老学到老。",

    # 较长句子 (6-10秒)
    "人工智能技术正在改变我们的生活方式。",
    "音乐是人类最古老也是最通用的语言形式。",
    "成功的道路从来都不是一帆风顺的。",
    "健康的身体是革命的本钱。",
    "知识就是力量，学习成就未来。",
    "每个人都是独一无二的存在。",
    "珍惜生命中的每一天。",
    "勇敢追求自己的梦想。",
    "微笑面对生活的挑战。",
    "时间管理是一门艺术。",
    "团队合作可以创造奇迹。",
    "创新是发展的第一动力。",
    "自律是成功的基石。",
    "感恩让生活更美好。",
    "坚持不懈才能取得胜利。",
    "健康饮食很重要。",
    "梦想是人生的动力。",
    "学习是终身的事业。",
    "良好的习惯受益终身。",
    "沟通是理解的桥梁。",

    # 绕口令 (训练发音)
    "八百标兵奔北坡，炮兵并排北边跑。",
    "黑化化肥会挥发，灰化化肥黑会发。",
    "红鲤鱼与绿鲤鱼与驴。",
    "牛郎恋刘娘，刘娘恋牛郎。",
    "四是四，十是十，十四是十四。",
    "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
    "石狮子咬死涩柿子，涩柿子咬死石狮子。",
    "大花碗里扣着大花活蛤蟆。",
    "喇嘛与哑巴，拉拉尼亚想喝哑巴水。",
    "南边来了个喇嘛，手里提着一只蛤蟆。",

    # 诗词经典
    "床前明月光，疑是地上霜。",
    "举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。",
    "夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。",
    "欲穷千里目，更上一层楼。",
    "红豆生南国，春来发几枝。",
    "愿君多采撷，此物最相思。",
    "独在异乡为异客，每逢佳节倍思亲。",
    "遥知兄弟登高处，遍插茱萸少一人。",

    # 日常对话
    "喂，你好，请问你找谁？",
    "不好意思，请问这是什么地方？",
    "麻烦您帮我看一下这个。",
    "请问您有什么需要帮助的吗？",
    "非常感谢您的帮助。",
    "不客气，这是我应该做的。",
    "对不起，给您添麻烦了。",
    "没关系，您别客气。",
    "您先请，我没关系。",
    "那就这样说定了。",

    # 故事叙述
    "从前有座山，山里有座庙。",
    "庙里有个老和尚讲故事。",
    "讲的什么故事呢。",
    "从前有座山。",
    "太阳从东方升起。",
    "月亮在夜空照耀。",
    "星星在天空中闪烁。",
    "小鸟在树枝上歌唱。",
    "小溪在山间流淌。",
    "花朵在春天开放。",

    # 情感表达
    "我真的很开心。",
    "我感到非常幸福。",
    "谢谢你一直陪着我。",
    "我真的很感动。",
    "心里有点难过。",
    "有点小失落。",
    "期待下一次的相遇。",
    "珍惜我们在一起的时光。",
    "永远都会支持你。",
    "你是最棒的。",

    # 鼓励语句
    "加油，你可以的。",
    "别放弃，坚持下去。",
    "相信自己，你很优秀。",
    "努力就会有收获。",
    "每天进步一点点。",
    "你一定行。",
    "不要害怕失败。",
    "勇敢迈出第一步。",
    "坚持下去就是胜利。",
    "梦想就在前方。",

    # 描述性语句
    "天空是蓝色的。",
    "白云在空中飘荡。",
    "太阳光很温暖。",
    "微风吹过脸庞。",
    "花儿散发着香味。",
    "树叶在风中摇摆。",
    "小鸟在空中飞翔。",
    "小河在流淌。",
    "星星在夜空闪烁。",
    "月亮又大又圆。",

    # 数字和时间
    "现在几点钟？",
    "今天几月几号？",
    "明天是星期几？",
    "一年有四季。",
    "一天有二十四个小时。",
    "一分钟有六十秒。",
    "春天百花盛开。",
    "夏天烈日炎炎。",
    "秋天硕果累累。",
    "冬天白雪皑皑。",

    # 动作指令
    "请把门关上。",
    "请把灯打开。",
    "请把水端过来。",
    "请把书递给我。",
    "请把椅子搬走。",
    "请把窗户打开。",
    "请把空调调低一点。",
    "请把音乐关掉。",
    "请把电视打开。",
    "请把空调关掉。",

    # 疑问句
    "这是真的吗？",
    "怎么会这样？",
    "为什么呢？",
    "你还好吗？",
    "你还好吗？",
    "你在做什么？",
    "你在哪里？",
    "你想要什么？",
    "你需要帮助吗？",
    "你想去哪里？",

    # 感叹句
    "太棒了！",
    "太好了！",
    "真美啊！",
    "好厉害！",
    "好可怕！",
    "好无聊啊！",
    "好热啊！",
    "好冷啊！",
    "好累啊！",
    "好开心！",

    # 更多日常
    "我正在听音乐。",
    "我正在看书。",
    "我正在做饭。",
    "我正在工作。",
    "我正在学习。",
    "我正在休息。",
    "我正在打电话。",
    "我正在发信息。",
    "我正在写东西。",
    "我正在打扫卫生。",

    # 复合长句
    "人工智能和机器学习正在各个领域发挥重要作用。",
    "语音合成技术已经取得了长足的进步。",
    "深度学习模型的能力越来越强大。",
    "语音识别技术在日常生活中越来越普及。",
    "自然语言处理让我们与机器的交互更自然。",
    "计算机视觉在自动驾驶中扮演关键角色。",
    "云计算为人工智能提供了强大的算力支持。",
    "大数据是人工智能发展的重要基础。",
    "边缘计算让AI设备更加智能和高效。",
    "5G网络的普及加速了AI应用的发展。",

    # RVC训练专用（高音/低音变化）
    "现在开始高音测试。",
    "接下来是低音测试。",
    "这是中音区域。",
    "请注意音高的变化。",
    "从低音升到高音。",
    "从高音降到低音。",
    "这一段保持高音。",
    "这一段保持低音。",
    "平滑过渡到高音。",
    "平滑过渡到低音。",
    "渐强。",
    "渐弱。",
    "保持节奏。",
    "放慢速度。",
    "加快速度。",
    "轻柔一些。",
    "用力一些。",
    "温柔一点。",
    "刚劲有力。",
    "悠扬婉转。",
]

# 每句话大约 3-5 秒，要达到 15-30 分钟需要 300-600 条
# 这里提供 300 条，组合使用

# Define FastAPI app
app = FastAPI()

# 挂载静态文件目录，用于提供转换后的音频下载
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class GUIConfig:
    def __init__(self) -> None:
        self.pth_path: str = ""
        self.index_path: str = ""
        self.pitch: int = 0
        self.formant: float = 0.0
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time: float = 2.5
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.rms_mix_rate: float = 0.0
        self.index_rate: float = 0.0
        self.n_cpu: int = 4
        self.f0method: str = "fcpe"
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""

class ConfigData(BaseModel):
    pth_path: str
    index_path: str
    sg_input_device: str
    sg_output_device: str
    threhold: int = -60
    pitch: int = 0
    formant: float = 0.0
    index_rate: float = 0.3
    rms_mix_rate: float = 0.0
    block_time: float = 0.25
    crossfade_length: float = 0.05
    extra_time: float = 2.5
    n_cpu: int = 4
    I_noise_reduce: bool = False
    O_noise_reduce: bool = False
    use_pv: bool = False
    f0method: str = "fcpe"


class RvcConvertRequest(BaseModel):
    input_url: str  # 音频URL
    model_name: str = "/home/RVC/assets/weights/mi-mengbao.pth"  # 模型路径
    f0method: str = "rmvpe"  # 基频检测方法
    index_rate: float = 0.7  # 检索强度
    index_path: str = ""  # 可选：index文件路径
    mix_audio_url: str = ""  # 可选：混合音频URL（背景音乐）
    # OSS 配置（可选，使用环境变量默认值）
    oss_bucket: str = "998555"  # OSS Bucket
    oss_endpoint: str = "oss-cn-beijing.aliyuncs.com"  # OSS Endpoint (华北2北京)
    oss_region: str = "cn-beijing"  # OSS Region


class GenerateTrainingDataRequest(BaseModel):
    """生成训练数据库请求"""
    user_id: str = ""  # 用户ID
    audio_url: str  # 参考音频URL (公网可访问)
    voice_prefix: str = "trainvoice"  # 音色前缀 (仅数字和小写字母，小于10个字符)
    target_duration_min: int = 15  # 目标时长 (分钟)，默认15分钟
    # output_dir 不再需要用户指定，自动生成


class TrainModelRequest(BaseModel):
    """训练模型请求"""
    user_id: str = ""  # 用户ID
    model_name: str  # 模型名称 (如 "my_voice")
    data_dir: str  # 训练数据目录 (包含 wav 和 txt 文件)
    sample_rate: int = 48000  # 采样率: 32000, 40000, 48000
    version: str = "v2"  # 版本: v1 或 v2
    epochs: int = 100  # 训练轮数
    batch_size: int = 4  # 批次大小
    gpu: str = "0"  # GPU编号

class Harvest(Process):
    def __init__(self, inp_q, opt_q):
        super(Harvest, self).__init__()
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld
        while True:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)


def phase_vocoder(a, b, fade_out, fade_in):
    """
    相位声码器，用于相位对齐
    源自: https://github.com/yxlllc/DDSP-SVC
    """
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


def _load_audio_file(path: str):
    """
    根据文件扩展名自动加载音频文件
    
    参数:
        path: 音频文件路径
        
    返回:
        AudioSegment: 加载的音频对象
    """
    from pydub import AudioSegment
    import os
    
    ext = os.path.splitext(path)[1].lower()
    format_map = {
        '.wav': 'wav',
        '.mp3': 'mp3',
        '.m4a': 'm4a',
        '.flac': 'flac',
        '.ogg': 'ogg',
    }
    format_str = format_map.get(ext)
    if format_str is None:
        raise ValueError(f"不支持的音频格式: {ext}")
    
    logger.info(f"加载音频: {os.path.basename(path)} ({ext})")
    return AudioSegment.from_file(path, format=format_str)


def simple_mix_audio(vocal_path: str, instrumental_path: str, output_path: str):
    """
    简单混合人声和背景音乐
    - 对齐时长（取较短长度）
    - 统一格式参数
    - 直接叠加混合
    
    参数:
        vocal_path: 人声文件路径
        instrumental_path: 背景音乐文件路径
        output_path: 输出文件路径
    """
    from pydub import AudioSegment
    
    # 读取音频
    vocal = _load_audio_file(vocal_path)
    instrumental = _load_audio_file(instrumental_path)
    
    logger.info(f"人声: {len(vocal)/1000:.2f}秒 | {vocal.frame_rate}Hz | {vocal.channels}通道")
    logger.info(f"背景音乐: {len(instrumental)/1000:.2f}秒 | {instrumental.frame_rate}Hz | {instrumental.channels}通道")
    
    # 对齐时长（取较短的长度，超出部分被截断）
    min_duration = min(len(vocal), len(instrumental))
    vocal = vocal[:min_duration]
    instrumental = instrumental[:min_duration]
    
    # 统一格式参数
    instrumental = instrumental.set_frame_rate(vocal.frame_rate)
    instrumental = instrumental.set_channels(vocal.channels)
    
    # 直接混合叠加
    mixed = vocal.overlay(instrumental)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 导出
    mixed.export(output_path, format='wav')
    logger.info(f"混合完成: {output_path}")
    logger.info(f"混合后: {len(mixed)/1000:.2f}秒 | {mixed.frame_rate}Hz")
    
    return mixed


def download_audio(url: str, save_path: str) -> str:
    """
    从URL下载音频文件
    
    参数:
        url: 音频文件URL
        save_path: 保存路径
        
    返回:
        str: 保存的文件路径
    """
    # 获取文件扩展名
    ext = url.split("?")[0].split(".")[-1].lower()
    if ext not in ["mp3", "wav", "m4a", "flac", "ogg"]:
        ext = "wav"
    
    # 确保扩展名正确
    if not save_path.endswith(f".{ext}"):
        save_path = f"{os.path.splitext(save_path)[0]}.{ext}"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    logger.info(f"下载音频: {url}")
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"下载失败: HTTP {response.status_code}")
    
    with open(save_path, "wb") as f:
        f.write(response.content)
    
    logger.info(f"已保存: {save_path}")
    return save_path


def generate_training_data_task(
    uid: str,
    user_id: str,
    audio_url: str,
    voice_prefix: str = "trainvoice",
    target_duration_min: int = 15,
    output_dir: str = "training_data_generated"
):
    """
    后台任务：生成训练数据库

    流程:
    1. 音色克隆 (阿里云 DashScope)
    2. 轮询等待音色就绪
    3. 批量生成语音

    参数:
        uid: 任务唯一ID
        user_id: 用户ID
        audio_url: 参考音频URL (公网可访问)
        voice_prefix: 音色前缀
        target_duration_min: 目标时长 (分钟)
        output_dir: 输出目录
    """
    import dashscope
    from dashscope.audio.tts_v2 import VoiceEnrollmentService, SpeechSynthesizer

    logger.info(f"[Task {uid}] 开始生成训练数据...")

    # 验证 audio_url
    if not audio_url:
        update_task_status(uid, TASK_STATUS["FAILED"], error_message="audio_url 不能为空")
        return

    # 验证 voice_prefix
    if not voice_prefix.replace("_", "").isalnum():
        update_task_status(uid, TASK_STATUS["FAILED"], error_message="voice_prefix 格式不正确")
        return
    if len(voice_prefix) > 10:
        update_task_status(uid, TASK_STATUS["FAILED"], error_message="voice_prefix 超过10个字符")
        return

    # 设置 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        update_task_status(uid, TASK_STATUS["FAILED"], error_message="未配置 DASHSCOPE_API_KEY")
        return

    dashscope.api_key = api_key

    logger.info(f"[Task {uid}] 参考音频: {audio_url}")
    logger.info(f"[Task {uid}] 音色前缀: {voice_prefix}")
    logger.info(f"[Task {uid}] 目标时长: {target_duration_min} 分钟")

    # ===== 步骤1: 复刻音色 =====
    logger.info(f"[Task {uid}] 步骤1: 开始复刻音色...")
    update_task_status(uid, TASK_STATUS["CLONING"])

    service = VoiceEnrollmentService()

    try:
        voice_id = service.create_voice(
            target_model="cosyvoice-v3-plus",
            prefix=voice_prefix,
            url=audio_url
        )
        logger.info(f"[Task {uid}] 音色复刻请求已提交, Voice ID: {voice_id}")
    except Exception as e:
        logger.error(f"[Task {uid}] 音色复刻失败: {e}")
        update_task_status(uid, TASK_STATUS["FAILED"], error_message=f"音色复刻失败: {str(e)}")
        return

    # ===== 步骤2: 轮询等待音色就绪 =====
    logger.info(f"[Task {uid}] 步骤2: 等待音色复刻完成...")

    max_attempts = 60  # 最多等待 10 分钟
    poll_interval = 10  # 每 10 秒查询一次

    for attempt in range(max_attempts):
        try:
            voice_info = service.query_voice(voice_id=voice_id)
            status = voice_info.get("status")
            logger.info(f"[Task {uid}] 尝试 {attempt + 1}/{max_attempts}: 状态 = '{status}'")

            if status == "OK":
                logger.info(f"[Task {uid}] 音色复刻成功!")
                break
            elif status == "UNDEPLOYED":
                raise RuntimeError(f"音色处理失败: {status}")
            elif status in ["PROCESSING", "DEPLOYING"]:
                time.sleep(poll_interval)
            else:
                logger.warning(f"[Task {uid}] 未知状态: {status}")
                time.sleep(poll_interval)
        except Exception as e:
            logger.warning(f"[Task {uid}] 查询失败: {e}")
            time.sleep(poll_interval)
    else:
        update_task_status(uid, TASK_STATUS["FAILED"], error_message="等待超时，音色尚未准备完成")
        return

    # 更新 voice_id 到数据库
    update_task_status(uid, TASK_STATUS["GENERATING"], voice_id=voice_id)

    # ===== 步骤3: 准备输出目录 =====
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[Task {uid}] 输出目录: {output_dir}")

    # ===== 步骤4: 批量生成训练音频 =====
    logger.info(f"[Task {uid}] 步骤3: 开始批量生成训练音频...")

    # 计算需要的文本数量 (每条约 3-5 秒)
    num_texts = target_duration_min * 20  # 约每3秒一条
    texts_to_generate = random.sample(TRAINING_TEXTS_POOL, min(num_texts, len(TRAINING_TEXTS_POOL)))

    # 如果不够，重复使用
    while len(texts_to_generate) < num_texts:
        texts_to_generate.extend(random.sample(TRAINING_TEXTS_POOL, min(num_texts - len(texts_to_generate), len(TRAINING_TEXTS_POOL))))

    logger.info(f"[Task {uid}] 将生成 {len(texts_to_generate)} 条文本")
    update_task_status(uid, TASK_STATUS["GENERATING"], total_texts=len(texts_to_generate))

    synthesizer = SpeechSynthesizer(model="cosyvoice-v3-plus", voice=voice_id)

    successful_count = 0
    failed_count = 0
    total_duration = 0.0

    for i, text in enumerate(texts_to_generate):
        output_file = os.path.join(output_dir, f"{i+1:04d}.wav")
        text_file = os.path.join(output_dir, f"{i+1:04d}.txt")

        try:
            # 生成音频
            audio_data = synthesizer.call(text)

            # 保存音频文件
            with open(output_file, "wb") as f:
                f.write(audio_data)

            # 保存对应的文本文件
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)

            successful_count += 1

            # 估算时长 (假设 40k 采样率，约 50KB/秒)
            duration = len(audio_data) / 50000
            total_duration += duration

            if (i + 1) % 10 == 0:
                logger.info(f"[Task {uid}] 已生成 {i+1}/{len(texts_to_generate)} 条，当前总时长约 {total_duration:.1f} 秒")

        except Exception as e:
            logger.error(f"[Task {uid}] 生成失败: {text[:20]}... - {e}")
            failed_count += 1

        # 避免请求过快
        time.sleep(0.5)

    logger.info(f"[Task {uid}] 生成完成! 成功: {successful_count}, 失败: {failed_count}")
    logger.info(f"[Task {uid}] 总时长约: {total_duration / 60:.1f} 分钟")

    # 更新最终状态
    update_task_status(
        uid,
        TASK_STATUS["COMPLETED"],
        successful_count=successful_count,
        failed_count=failed_count,
        estimated_duration_min=total_duration / 60
    )

    logger.info(f"[Task {uid}] 任务完成!")


def run_training_task(
    uid: str,
    user_id: str,
    model_name: str,
    data_dir: str,
    sample_rate: int = 48000,
    version: str = "v2",
    epochs: int = 100,
    batch_size: int = 4,
    gpu: str = "0"
):
    """
    后台任务：训练RVC模型

    流程:
    1. 预处理数据 (切片、重采样)
    2. 提取特征 (Hubert + F0)
    3. 训练模型

    参数:
        uid: 任务唯一ID
        user_id: 用户ID
        model_name: 模型名称
        data_dir: 训练数据目录
        sample_rate: 采样率
        version: 版本 (v1/v2)
        epochs: 训练轮数
        batch_size: 批次大小
        gpu: GPU编号
    """
    import subprocess
    import shutil

    logger.info(f"[Train Task {uid}] 开始训练模型: {model_name}")

    # 验证数据目录
    if not os.path.exists(data_dir):
        update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                error_message=f"数据目录不存在: {data_dir}")
        return

    # 检查数据目录中是否有音频文件
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    if len(wav_files) == 0:
        update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                error_message="数据目录中没有音频文件")
        return

    logger.info(f"[Train Task {uid}] 找到 {len(wav_files)} 个音频文件")

    # 创建实验目录
    exp_dir = os.path.join("logs", model_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 保存训练数据链接
    try:
        # 创建 filelist.txt
        filelist_path = os.path.join(exp_dir, "filelist.txt")
        with open(filelist_path, "w", encoding="utf-8") as f:
            for wav_file in wav_files:
                wav_path = os.path.join(data_dir, wav_file)
                txt_file = wav_file.replace(".wav", ".txt")
                txt_path = os.path.join(data_dir, txt_file)

                # 尝试读取对应的文本
                text = ""
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        text = tf.read().strip()

                if text:
                    f.write(f"{wav_path}|{text}\n")
                else:
                    f.write(f"{wav_path}\n")

        logger.info(f"[Train Task {uid}] 已创建 filelist.txt")
    except Exception as e:
        logger.error(f"[Train Task {uid}] 创建 filelist 失败: {e}")
        update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                error_message=f"创建filelist失败: {str(e)}")
        return

    # ===== 步骤1: 预处理 =====
    logger.info(f"[Train Task {uid}] 步骤1: 预处理数据...")
    update_model_train_status(uid, MODEL_TRAIN_STATUS["PREPROCESSING"])

    try:
        # 调用预处理脚本
        # python infer/modules/train/preprocess.py <data_dir> <sr> <n_p> <exp_dir> <noparallel> <per>
        preprocess_cmd = [
            sys.executable,
            "infer/modules/train/preprocess.py",
            data_dir,
            str(sample_rate),
            "3.7",
            exp_dir,
            "False",
            "3.7"
        ]

        logger.info(f"[Train Task {uid}] 执行预处理: {' '.join(preprocess_cmd)}")
        result = subprocess.run(preprocess_cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"[Train Task {uid}] 预处理失败: {result.stderr}")
            update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                    error_message=f"预处理失败: {result.stderr[:500]}")
            return
        logger.info(f"[Train Task {uid}] 预处理完成")
    except Exception as e:
        logger.error(f"[Train Task {uid}] 预处理异常: {e}")
        update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                error_message=f"预处理异常: {str(e)}")
        return

    # ===== 步骤2: 提取特征 =====
    logger.info(f"[Train Task {uid}] 步骤2: 提取特征...")
    update_model_train_status(uid, MODEL_TRAIN_STATUS["EXTRACTING"])

    try:
        # 调用特征提取脚本
        # python infer/modules/train/extract_feature_print.py <device> <n_part> <i_part> <exp_dir> <version> <is_half>
        extract_cmd = [
            sys.executable,
            "infer/modules/train/extract_feature_print.py",
            gpu,
            "1",
            "0",
            exp_dir,
            version,
            "True"  # is_half
        ]

        logger.info(f"[Train Task {uid}] 执行特征提取: {' '.join(extract_cmd)}")
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            logger.error(f"[Train Task {uid}] 特征提取失败: {result.stderr}")
            update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                    error_message=f"特征提取失败: {result.stderr[:500]}")
            return
        logger.info(f"[Train Task {uid}] 特征提取完成")
    except Exception as e:
        logger.error(f"[Train Task {uid}] 特征提取异常: {e}")
        update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                error_message=f"特征提取异常: {str(e)}")
        return

    # ===== 步骤3: 训练模型 =====
    logger.info(f"[Train Task {uid}] 步骤3: 开始训练...")
    update_model_train_status(uid, MODEL_TRAIN_STATUS["TRAINING"], total_epochs=epochs)

    try:
        # 复制配置文件到实验目录
        config_src = f"configs/{version}/{sample_rate//1000}k.json"
        config_dst = os.path.join(exp_dir, "config.json")

        if os.path.exists(config_src):
            shutil.copy(config_src, config_dst)
            logger.info(f"[Train Task {uid}] 已复制配置文件")

        # 调用训练脚本
        # python infer/modules/train/train.py -e <exp_name> -g <gpu> -pg <pretrainG> -pd <pretrainD>
        train_cmd = [
            sys.executable,
            "infer/modules/train/train.py",
            "-e", model_name,
            "-g", gpu,
            "-pg", "",  # 可选：预训练模型
            "-pd", ""
        ]

        logger.info(f"[Train Task {uid}] 开始训练模型: {' '.join(train_cmd)}")

        # 训练过程需要较长时间，使用子进程监控
        process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        current_epoch = 0
        loss_g = 0.0
        loss_d = 0.0

        # 监控训练输出
        while True:
            # 检查进程是否结束
            retcode = process.poll()
            if retcode is not None:
                # 进程结束
                if retcode == 0:
                    logger.info(f"[Train Task {uid}] 训练完成!")
                else:
                    stderr = process.stderr.read() if process.stderr else ""
                    logger.error(f"[Train Task {uid}] 训练失败: {stderr}")
                    update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                            error_message=f"训练失败: {stderr[:500]}")
                break

            # 读取输出
            try:
                line = process.stdout.readline()
                if line:
                    # 解析训练输出，提取epoch信息
                    # 示例输出: "Step: 100/1000 | Loss_G: 1.234 | Loss_D: 0.567"
                    if "Step:" in line or "Epoch:" in line:
                        logger.info(f"[Train Task {uid}] {line.strip()}")

                        # 尝试提取epoch
                        import re
                        epoch_match = re.search(r'Epoch[:\s]+(\d+)', line)
                        if epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            update_model_train_status(uid, MODEL_TRAIN_STATUS["TRAINING"],
                                                    current_epoch=current_epoch,
                                                    loss_g=loss_g, loss_d=loss_d)

                        # 尝试提取loss
                        loss_g_match = re.search(r'Loss_G[:\s]+([\d.]+)', line)
                        loss_d_match = re.search(r'Loss_D[:\s]+([\d.]+)', line)
                        if loss_g_match:
                            loss_g = float(loss_g_match.group(1))
                        if loss_d_match:
                            loss_d = float(loss_d_match.group(1))
            except:
                pass

            time.sleep(1)

        # 如果进程正常结束
        if process.returncode == 0:
            # 查找生成的模型文件
            model_path = os.path.join(exp_dir, "G_latest.pth")
            if os.path.exists(model_path):
                # 复制到 weights 目录
                weights_dir = "assets/weights"
                os.makedirs(weights_dir, exist_ok=True)
                final_model_path = os.path.join(weights_dir, f"{model_name}.pth")
                shutil.copy(model_path, final_model_path)
                logger.info(f"[Train Task {uid}] 模型已保存到: {final_model_path}")

                update_model_train_status(uid, MODEL_TRAIN_STATUS["COMPLETED"],
                                        current_epoch=epochs,
                                        model_path=final_model_path)
            else:
                update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                        error_message="未找到生成的模型文件")

    except Exception as e:
        logger.error(f"[Train Task {uid}] 训练异常: {e}")
        update_model_train_status(uid, MODEL_TRAIN_STATUS["FAILED"],
                                error_message=f"训练异常: {str(e)}")

    logger.info(f"[Train Task {uid}] 训练任务结束")


class AudioAPI:
    def __init__(self) -> None:
        self.gui_config = GUIConfig()
        self.config = None  # Initialize Config object as None
        self.flag_vc = False
        self.function = "vc"
        self.delay_time = 0
        self.rvc = None  # Initialize RVC object as None
        self.inp_q = None
        self.opt_q = None
        self.n_cpu = min(cpu_count(), 8)

    def initialize_queues(self):
        self.inp_q = Queue()
        self.opt_q = Queue()
        for _ in range(self.n_cpu):
            p = Harvest(self.inp_q, self.opt_q)
            p.daemon = True
            p.start()

    def load(self):
        input_devices, output_devices, _, _ = self.get_devices()
        try:
            with open("configs/config.json", "r", encoding='utf-8') as j:
                data = json.load(j)
                if data["sg_input_device"] not in input_devices:
                    data["sg_input_device"] = input_devices[sd.default.device[0]]
                if data["sg_output_device"] not in output_devices:
                    data["sg_output_device"] = output_devices[sd.default.device[1]]
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            with open("configs/config.json", "w", encoding='utf-8') as j:
                data = {
                    "pth_path": "",
                    "index_path": "",
                    "sg_input_device": input_devices[sd.default.device[0]],
                    "sg_output_device": output_devices[sd.default.device[1]],
                    "threhold": -60,
                    "pitch": 0,
                    "formant": 0.0,
                    "index_rate": 0,
                    "rms_mix_rate": 0,
                    "block_time": 0.25,
                    "crossfade_length": 0.05,
                    "extra_time": 2.5,
                    "n_cpu": 4,
                    "f0method": "fcpe",
                    "use_jit": False,
                    "use_pv": False,
                }
                json.dump(data, j, ensure_ascii=False)
        return data

    def set_values(self, values):
        logger.info(f"Setting values: {values}")
        if not values.pth_path.strip():
            raise HTTPException(status_code=400, detail="Please select a .pth file")
        if not values.index_path.strip():
            raise HTTPException(status_code=400, detail="Please select an index file")
        self.set_devices(values.sg_input_device, values.sg_output_device)
        # 确保 config 已初始化
        if self.config is not None:
            self.config.use_jit = False
        self.gui_config.pth_path = values.pth_path
        self.gui_config.index_path = values.index_path
        self.gui_config.threhold = values.threhold
        self.gui_config.pitch = values.pitch
        self.gui_config.formant = values.formant
        self.gui_config.block_time = values.block_time
        self.gui_config.crossfade_time = values.crossfade_length
        self.gui_config.extra_time = values.extra_time
        self.gui_config.I_noise_reduce = values.I_noise_reduce
        self.gui_config.O_noise_reduce = values.O_noise_reduce
        self.gui_config.rms_mix_rate = values.rms_mix_rate
        self.gui_config.index_rate = values.index_rate
        self.gui_config.n_cpu = values.n_cpu
        self.gui_config.use_pv = values.use_pv
        self.gui_config.f0method = values.f0method
        return True

    def start_vc(self):
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.rvc = rvc_for_realtime.RVC(
            self.gui_config.pitch,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            self.rvc if self.rvc else None,
        )
        self.gui_config.samplerate = (
            self.rvc.tgt_sr
            if self.gui_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)
        thread_vc = threading.Thread(target=self.soundinput)
        thread_vc.start()

    def soundinput(self):
        channels = 1 if sys.platform == "darwin" else 2
        with sd.Stream(
            channels=channels,
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.gui_config.samplerate,
            dtype="float32",
        ) as stream:
            global stream_latency
            stream_latency = stream.latency[-1]
            while self.flag_vc:
                time.sleep(self.gui_config.block_time)
                logger.info("Audio block passed.")
        logger.info("Ending VC")

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.gui_config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        # input noise reduction and resampling
        if self.gui_config.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[160:]
            )
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.gui_config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        # output noise reduction
        if self.gui_config.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
        # volume envelop mixing
        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            if self.gui_config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        logger.info(f"sola_offset = {sola_offset}")
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        if sys.platform == "darwin":
            outdata[:] = infer_wav[: self.block_frame].cpu().numpy()[:, np.newaxis]
        else:
            outdata[:] = infer_wav[: self.block_frame].repeat(2, 1).t().cpu().numpy()
        total_time = time.perf_counter() - start_time
        logger.info(f"Infer time: {total_time:.2f}")

    def get_devices(self, update: bool = True):
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0
        ]
        return (
            input_devices,
            output_devices,
            input_devices_indices,
            output_devices_indices,
        )

    def set_devices(self, input_device, output_device):
        (
            input_devices,
            output_devices,
            input_device_indices,
            output_device_indices,
        ) = self.get_devices()
        logger.debug(f"Available input devices: {input_devices}")
        logger.debug(f"Available output devices: {output_devices}")
        logger.debug(f"Selected input device: {input_device}")
        logger.debug(f"Selected output device: {output_device}")

        if input_device not in input_devices:
            logger.error(f"Input device '{input_device}' is not in the list of available devices")
            raise HTTPException(status_code=400, detail=f"Input device '{input_device}' is not available")
        
        if output_device not in output_devices:
            logger.error(f"Output device '{output_device}' is not in the list of available devices")
            raise HTTPException(status_code=400, detail=f"Output device '{output_device}' is not available")

        sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
        sd.default.device[1] = output_device_indices[output_devices.index(output_device)]
        logger.info(f"Input device set to {sd.default.device[0]}: {input_device}")
        logger.info(f"Output device set to {sd.default.device[1]}: {output_device}")

audio_api = AudioAPI()

@app.get("/inputDevices", response_model=list)
def get_input_devices():
    try:
        input_devices, _, _, _ = audio_api.get_devices()
        return input_devices
    except Exception as e:
        logger.error(f"Failed to get input devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get input devices")

@app.get("/outputDevices", response_model=list)
def get_output_devices():
    try:
        _, output_devices, _, _ = audio_api.get_devices()
        return output_devices
    except Exception as e:
        logger.error(f"Failed to get output devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get output devices")

@app.post("/config")
def configure_audio(config_data: ConfigData):
    try:
        logger.info(f"Configuring audio with data: {config_data}")
        if audio_api.set_values(config_data):
            settings = config_data.dict()
            settings["use_jit"] = False
            with open("configs/config.json", "w", encoding='utf-8') as j:
                json.dump(settings, j, ensure_ascii=False)
            logger.info("Configuration set successfully")
            return {"message": "Configuration set successfully"}
    except HTTPException as e:
        logger.error(f"Configuration error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration failed: {e}")

@app.post("/start")
def start_conversion():
    try:
        if not audio_api.flag_vc:
            audio_api.start_vc()
            return {"message": "Audio conversion started"}
        else:
            logger.warning("Audio conversion already running")
            raise HTTPException(status_code=400, detail="Audio conversion already running")
    except HTTPException as e:
        logger.error(f"Start conversion error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Failed to start conversion: {e}")
        raise HTTPException(status_code=500, detail="Failed to start conversion: {e}")

@app.post("/stop")
def stop_conversion():
    try:
        if audio_api.flag_vc:
            audio_api.flag_vc = False
            global stream_latency
            stream_latency = -1
            return {"message": "Audio conversion stopped"}
        else:
            logger.warning("Audio conversion not running")
            raise HTTPException(status_code=400, detail="Audio conversion not running")
    except HTTPException as e:
        logger.error(f"Stop conversion error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Failed to stop conversion: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop conversion: {e}")


@app.post("/convert/url")
def convert_url(request: RvcConvertRequest):
    """
    根据URL进行RVC转换
    - 下载音频文件
    - 使用指定模型进行转换
    - 可选：混合背景音乐
    - 上传到阿里云OSS
    - 返回OSS文件URL
    """
    from fastapi import Request
    try:
        from scipy.io import wavfile
        from infer.modules.vc.modules import VC
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"收到转换请求: input_url={request.input_url}, model_name={request.model_name}")
        logger.info(f"混合音频URL: {request.mix_audio_url if request.mix_audio_url else '未提供'}")
        
        # ==================== 1. 下载输入音频 ====================
        input_path = download_audio(request.input_url, os.path.join("cache", f"input_{timestamp}"))
        
        # ==================== 2. RVC转换 ====================
        rvc_output_filename = f"rvc_{timestamp}.wav"
        rvc_output_path = os.path.join("static", rvc_output_filename)
        os.makedirs("static", exist_ok=True)
        
        logger.info("加载VC模型...")
        config = Config()
        vc = VC(config)
        vc.get_vc(request.model_name)
        
        logger.info(f"开始转换: f0method={request.f0method}, index_rate={request.index_rate}")
        _, wav_opt = vc.vc_single(
            0,
            input_path,
            0,  # f0up_key
            None,
            request.f0method,
            request.index_path if request.index_path else None,
            None,
            request.index_rate,
            3,  # filter_radius
            0,  # resample_sr
            1,  # rms_mix_rate
            0.33,  # protect
        )
        
        # 保存RVC转换后的音频
        wavfile.write(rvc_output_path, wav_opt[0], wav_opt[1])
        logger.info(f"RVC转换完成: {rvc_output_path}")
        
        # 清理输入文件
        if os.path.exists(input_path):
            os.remove(input_path)
        
        # ==================== 3. 混合背景音乐（如果提供） ====================
        final_output_path = rvc_output_path
        is_mixed = False
        
        if request.mix_audio_url:
            logger.info("开始混合背景音乐...")
            
            # 下载背景音乐
            instrumental_path = download_audio(
                request.mix_audio_url, 
                os.path.join("cache", f"instrumental_{timestamp}")
            )
            
            # 混合音频
            mixed_filename = f"mixed_{timestamp}.wav"
            mixed_output_path = os.path.join("static", mixed_filename)
            
            simple_mix_audio(
                vocal_path=rvc_output_path,
                instrumental_path=instrumental_path,
                output_path=mixed_output_path
            )
            
            # 清理背景音乐临时文件
            if os.path.exists(instrumental_path):
                os.remove(instrumental_path)
            
            # 清理单独的RVC输出文件
            if os.path.exists(rvc_output_path):
                os.remove(rvc_output_path)
            
            final_output_path = mixed_output_path
            is_mixed = True
            logger.info("音频混合完成")
        
        # ==================== 4. 上传到阿里云OSS ====================
        oss_config = oss.config.load_default()
        oss_config.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        
        # 从请求参数或环境变量读取OSS配置
        oss_region = request.oss_region or os.environ.get("OSS_REGION", "cn-hangzhou")
        oss_endpoint = request.oss_endpoint or os.environ.get("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
        oss_bucket = request.oss_bucket or os.environ.get("OSS_BUCKET", "your-bucket-name")
        
        oss_config.region = oss_region
        oss_config.endpoint = oss_endpoint
        
        oss_client = oss.Client(oss_config)
        
        # 生成OSS key
        oss_key = f"rvc_output/{os.path.basename(final_output_path)}"
        
        logger.info(f"上传到OSS: bucket={oss_bucket}, key={oss_key}")
        
        # 分片上传
        file_size = os.path.getsize(final_output_path)
        part_size = 5 * 1024 * 1024  # 每个分片5MB
        part_number = 1
        upload_parts = []
        offset = 0
        
        # 初始化分片上传
        initiate_result = oss_client.initiate_multipart_upload(
            oss.InitiateMultipartUploadRequest(
                bucket=oss_bucket,
                key=oss_key
            ))
        upload_id = initiate_result.upload_id
        
        with open(final_output_path, 'rb') as f:
            while offset < file_size:
                current_part_size = min(part_size, file_size - offset)
                f.seek(offset)
                part_data = f.read(current_part_size)
                
                part_result = oss_client.upload_part(
                    oss.UploadPartRequest(
                        bucket=oss_bucket,
                        key=oss_key,
                        upload_id=upload_id,
                        part_number=part_number,
                        body=part_data
                    ))
                
                upload_parts.append(oss.UploadPart(
                    part_number=part_number,
                    etag=part_result.etag
                ))
                
                offset += current_part_size
                part_number += 1
        
        # 完成分片上传
        upload_parts.sort(key=lambda p: p.part_number)
        oss_client.complete_multipart_upload(
            oss.CompleteMultipartUploadRequest(
                bucket=oss_bucket,
                key=oss_key,
                upload_id=upload_id,
                complete_multipart_upload=oss.CompleteMultipartUpload(parts=upload_parts)
            ))
        
        oss_url = f"https://{oss_bucket}.{oss_endpoint}/{oss_key}"
        logger.info(f"上传成功: {oss_url}")
        
        # ==================== 5. 清理本地文件 ====================
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        
        # ==================== 6. 返回结果 ====================
        return {
            "message": "转换成功" + ("（已混合背景音乐）" if is_mixed else ""),
            "output_url": oss_url,
            "is_mixed": is_mixed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转换失败: {e}")
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


@app.post("/train/generate")
def generate_training_db(request: GenerateTrainingDataRequest):
    """
    生成训练数据库 (异步)

    流程:
    1. 使用阿里云语音克隆服务复刻音色
    2. 轮询等待音色就绪
    3. 批量生成训练语音

    参数:
        user_id: 用户ID
        audio_url: 参考音频URL (公网可访问)
        voice_prefix: 音色前缀 (仅数字和小写字母，小于10个字符)
        target_duration_min: 目标时长 (分钟)，默认15分钟

    返回:
        任务ID (uid) 和 output_dir，可用于查询状态
    """
    try:
        logger.info(f"收到训练数据生成请求: audio_url={request.audio_url}")
        logger.info(f"参数: user_id={request.user_id}, voice_prefix={request.voice_prefix}, target={request.target_duration_min}min")

        # 验证参数
        if not request.audio_url:
            raise HTTPException(status_code=400, detail="audio_url 不能为空")

        # 生成唯一的 output_dir (基于 user_id 和 voice_prefix + 时间戳)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("training_data", request.user_id or "default",
                                  f"{request.voice_prefix}_{timestamp}")

        # 生成任务ID
        uid = str(uuid.uuid4())

        # 创建任务记录
        create_training_task(
            uid=uid,
            user_id=request.user_id,
            audio_url=request.audio_url,
            voice_prefix=request.voice_prefix,
            target_duration_min=request.target_duration_min,
            output_dir=output_dir
        )

        # 启动后台线程执行任务
        thread = threading.Thread(
            target=generate_training_data_task,
            args=(
                uid,
                request.user_id,
                request.audio_url,
                request.voice_prefix,
                request.target_duration_min,
                output_dir
            )
        )
        thread.daemon = True
        thread.start()

        logger.info(f"任务已创建: uid={uid}")

        return {
            "uid": uid,
            "output_dir": output_dir,
            "message": "任务已创建，请使用 uid 查询状态"
        }

    except Exception as e:
        logger.error(f"创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@app.get("/train/status/{uid}")
def get_training_status(uid: str):
    """
    查询训练数据生成状态

    参数:
        uid: 任务ID

    返回:
        任务状态信息
    """
    try:
        task = get_task_status(uid)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {uid}")

        return task

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询任务状态失败: {str(e)}")


@app.post("/train/model")
def train_model(request: TrainModelRequest):
    """
    训练RVC模型 (异步)

    流程:
    1. 预处理数据 (切片、重采样)
    2. 提取特征 (Hubert + F0)
    3. 训练模型

    参数:
        user_id: 用户ID
        model_name: 模型名称 (如 "my_voice")
        data_dir: 训练数据目录 (包含 wav 和 txt 文件)
        sample_rate: 采样率 (32000/40000/48000)
        version: 版本 (v1 或 v2)
        epochs: 训练轮数 (默认100)
        batch_size: 批次大小 (默认4)
        gpu: GPU编号 (默认0)

    返回:
        任务ID (uid)，可用于查询状态
    """
    try:
        logger.info(f"收到模型训练请求: model_name={request.model_name}")
        logger.info(f"参数: user_id={request.user_id}, data_dir={request.data_dir}")
        logger.info(f"参数: sample_rate={request.sample_rate}, version={request.version}, epochs={request.epochs}")

        # 验证参数
        if not request.model_name:
            raise HTTPException(status_code=400, detail="模型名称不能为空")

        if not request.data_dir:
            raise HTTPException(status_code=400, detail="训练数据目录不能为空")

        if request.sample_rate not in [32000, 40000, 48000]:
            raise HTTPException(status_code=400, detail="采样率必须是 32000, 40000 或 48000")

        if request.version not in ["v1", "v2"]:
            raise HTTPException(status_code=400, detail="版本必须是 v1 或 v2")

        # 生成任务ID
        uid = str(uuid.uuid4())

        # 创建任务记录
        create_model_train_task(
            uid=uid,
            user_id=request.user_id,
            model_name=request.model_name,
            data_dir=request.data_dir,
            sample_rate=request.sample_rate,
            version=request.version,
            epochs=request.epochs,
            batch_size=request.batch_size,
            gpu=request.gpu
        )

        # 启动后台线程执行训练
        thread = threading.Thread(
            target=run_training_task,
            args=(
                uid,
                request.user_id,
                request.model_name,
                request.data_dir,
                request.sample_rate,
                request.version,
                request.epochs,
                request.batch_size,
                request.gpu
            )
        )
        thread.daemon = True
        thread.start()

        logger.info(f"训练任务已创建: uid={uid}")

        return {
            "uid": uid,
            "message": "训练任务已创建，请使用 uid 查询状态"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建训练任务失败: {str(e)}")


@app.get("/train/model/status/{uid}")
def get_model_train_status_api(uid: str):
    """
    查询模型训练状态

    参数:
        uid: 任务ID

    返回:
        训练任务状态信息
    """
    try:
        task = get_model_train_status(uid)

        if not task:
            raise HTTPException(status_code=404, detail=f"训练任务不存在: {uid}")

        return task

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询训练状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询训练状态失败: {str(e)}")


@app.get("/train/tasks/{user_id}")
def get_user_tasks(user_id: str):
    """
    查询用户的所有任务

    参数:
        user_id: 用户ID

    返回:
        用户的所有生成任务和训练任务
    """
    try:
        # 获取生成任务
        generate_tasks = get_user_training_tasks(user_id)

        # 获取训练任务
        train_tasks = get_user_model_train_tasks(user_id)

        return {
            "user_id": user_id,
            "generate_tasks": generate_tasks,
            "train_tasks": train_tasks,
            "generate_task_count": len(generate_tasks),
            "train_task_count": len(train_tasks)
        }

    except Exception as e:
        logger.error(f"查询用户任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询用户任务失败: {str(e)}")

if __name__ == "__main__":
    if sys.platform == "win32":
        freeze_support()
    load_dotenv()
    os.environ["OMP_NUM_THREADS"] = "4"
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    from tools.torchgate import TorchGate
    import tools.rvc_for_realtime as rvc_for_realtime
    from configs.config import Config
    audio_api.config = Config()
    audio_api.initialize_queues()
    uvicorn.run(app, host="0.0.0.0", port=8061)
