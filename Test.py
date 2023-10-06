import cv2
import numpy as np
import matplotlib.pyplot as plt

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
corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.1, minDistance=10)
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

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    # 合并光流轨迹和当前帧
    img = cv2.add(frame, mask)

    # 计算并记录振幅
    amplitude = np.max(np.linalg.norm(good_new - good_old, axis=1))
    amplitudes.append(amplitude)

    # 显示结果
    cv2.imshow('Frame', img)

    # 更新prev_gray和特征点
    prev_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 计算当前帧数
    frame_count += 1

    # 检查是否播放完毕
    if frame_count == total_frames:
        break


    # 按Esc键退出
    if cv2.waitKey(30) & 0xff == 27:
        break

# 释放资源
video_capture.release()
cv2.destroyAllWindows()

# 绘制振幅随时间变化的折线图
plt.plot(range(len(amplitudes)), amplitudes)
plt.xlabel('帧数')
plt.ylabel('振幅')
plt.title('特征点振幅随时间变化')
plt.show()