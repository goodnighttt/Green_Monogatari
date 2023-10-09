import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine

# 打开视频文件
video_capture = cv2.VideoCapture('2.mp4')

# 获取视频的总帧数
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始化Lucas-Kanade光流法参数
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 创建随机颜色用于绘制跟踪点
color = np.random.randint(0, 255, (100, 3))

# 读取第一帧
ret, prev_frame = video_capture.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 使用Shi-Tomasi角点检测获取初始的特征点位置
# 调整qualityLevel以筛选出质量较高的特征点，范围为0到1
# 调整minDistance以设置特征点之间的最小距离
corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
p0 = corners.reshape(-1, 1, 2)

frame_count = 0  # 用于计算当前帧数

# 定义一个空的mask用于绘制轨迹
mask = np.zeros_like(prev_frame)

# 初始化存储特征点振幅的列表
amplitudes = []

while True:
    # 读取当前帧
    ret, frame = video_capture.read()
    if not ret:
        break

    # 将当前帧转换为灰度图像
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用光流法追踪特征点
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

    # 选择良好的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 计算并记录振幅
    amplitude = np.max(np.linalg.norm(good_new - good_old, axis=1))
    amplitudes.append(amplitude)

    # 更新prev_gray和特征点
    prev_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 计算当前帧数
    frame_count += 1

    # 检查是否播放完毕
    if frame_count == total_frames:
        break

print(amplitudes)

# 释放资源
video_capture.release()
cv2.destroyAllWindows()

sns.set()
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示，必须放在sns.set之后

def normalize_array(arr):
    # 找到数组的最小值和最大值
    min_val = min(arr)
    max_val = max(arr)

    # 归一化数组
    normalized_arr = [(x - min_val) / (max_val - min_val) for x in arr]

    return normalized_arr

def map_to_seven_intervals(normalized_arr):
    # 将归一化数组映射到七个区间
    num_intervals = 7
    interval_size = len(normalized_arr) // num_intervals
    mapped_intervals = []

    for i in range(0, len(normalized_arr), interval_size):
        interval = normalized_arr[i:i+interval_size]
        avg_value = sum(interval) / len(interval)
        mapped_intervals.append(avg_value)

    return mapped_intervals

def generate_music_from_intervals(intervals):
    # 创建一个音乐序列
    song = AudioSegment.empty()

    # 音符持续时间（毫秒）
    note_duration = 1000

    for interval in intervals:
        # 计算音符持续时间的帧数
        num_frames = int(note_duration * 44100 / 1000)  # 采样率为44100

        # 生成一个音符的numpy数组，这里假设使用基础频率440Hz
        t = np.linspace(0, note_duration / 1000, num_frames)
        note_waveform = np.sin(2 * np.pi * 440 * interval * t)

        # 将numpy数组转换为AudioSegment对象，并添加到音乐中
        note = AudioSegment(
            note_waveform.tobytes(),
            frame_rate=44100,  # 采样率
            sample_width=2,  # 样本宽度设置为2，表示16位样本宽度
            channels=1  # 单声道
        )
        song += note

    # 保存生成的音乐
    song.export("generated_music.mp3", format="mp3")

# 归一化数组
normalized_array = normalize_array(amplitudes)

# 映射到七个区间
mapped_intervals = map_to_seven_intervals(normalized_array)

# 生成音乐
generate_music_from_intervals(mapped_intervals)