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

# 打印振幅数据
print(amplitudes)

# 释放资源
video_capture.release()
cv2.destroyAllWindows()

sns.set()
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示，必须放在sns.set之后

# 定义音高频率映射
note_frequencies = {
    "do": 261.6,
    "re": 293.6,
    "mi": 329.6,
    "fa": 349.2,
    "so": 392.0,
    "la": 440.0,
    "xi": 493.8,
}

# 将浮点数据映射到0-7的整数区间
min_value = min(amplitudes)
max_value = max(amplitudes)
mapped_data = ((np.array(amplitudes) - min_value) / (max_value - min_value) * 6).astype(int)

print(mapped_data)

# 定义音符持续时间（以毫秒为单位）
note_duration = 1000

# 创建一个空的音频段
music = AudioSegment.empty()

# 将整数数据映射到音高频率并添加到音乐中
for note_index in mapped_data:
    note_name = list(note_frequencies.keys())[note_index]
    frequency = note_frequencies[note_name]

    # 生成正弦波音频
    t = np.linspace(0, note_duration / 1000, int(44100 * (note_duration / 1000)), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    sine_wave = (sine_wave * 32767).astype(np.int16)  # 转换为16位整数

    # 创建pydub音频段并添加到音乐中
    note_audio = AudioSegment(sine_wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    music += note_audio

# 保存音乐到文件（可选）
music.export("output_music.wav", format="wav")

