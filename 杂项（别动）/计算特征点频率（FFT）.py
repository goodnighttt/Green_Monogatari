import cv2
import numpy as np
import matplotlib.pyplot as plt

# 打开视频文件
video_capture = cv2.VideoCapture('2.mp4')

# 获取视频的帧率
fps = video_capture.get(cv2.CAP_PROP_FPS)

# 初始化Lucas-Kanade光流法参数
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 读取第一帧
ret, prev_frame = video_capture.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 使用Shi-Tomasi角点检测获取初始的特征点位置
corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
p0 = corners.reshape(-1, 1, 2)

frame_count = 0  # 用于计算当前帧数

# 初始化存储特征点位置的列表
feature_point_positions = []

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

    # 计算特征点的位移
    displacement = np.linalg.norm(good_new - good_old, axis=1)

    # 计算时间点
    time_point = frame_count / fps

    # 存储特征点位置和时间点
    feature_point_positions.append((time_point, displacement))

    # 更新prev_gray和特征点
    prev_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 计算当前帧数
    frame_count += 1

# 释放资源
video_capture.release()
cv2.destroyAllWindows()

# 处理特征点位置数据以计算振动频率
feature_point_positions = np.array(feature_point_positions)
time_points = feature_point_positions[:, 0]
displacements = feature_point_positions[:, 1]

# 计算FFT并获得振动频率
fft_result = np.fft.fft(displacements)
frequency = np.fft.fftfreq(len(fft_result), 1 / fps)
amplitudes = np.abs(fft_result)
positive_freq_indices = np.where(frequency > 0)
frequency = frequency[positive_freq_indices]
amplitudes = amplitudes[positive_freq_indices]
main_frequencies = frequency[np.argmax(amplitudes, axis=1)]

# 绘制振动频率随时间变化的折线图
plt.plot(time_points, main_frequencies)
plt.xlabel('时间 (秒)')
plt.ylabel('振动频率 (Hz)')
plt.title('特征点振动频率随时间变化')
plt.show()
