from pydub import AudioSegment
import numpy as np

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

# 示例输入数组，每个数字代表一个音调
input_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 音符持续时间（毫秒）
note_duration = 1000

def generate_music_from_array(arr):
    # 创建一个音乐序列
    song = AudioSegment.empty()

    for value in arr:
        # 根据数组中的值来生成音符的频率
        frequency = int(261.63 * 2 ** ((value - 1) / 12))  # 261.63Hz对应C4，每半音阶增加2 ** (1/12) 倍

        # 生成一个音符的numpy数组
        t = np.linspace(0, note_duration / 1000, int(note_duration * 44100 / 1000))
        note_waveform = np.sin(2 * np.pi * frequency * t)

        # 将numpy数组转换为AudioSegment对象，并添加到音乐中
        note = AudioSegment(
            note_waveform.tobytes(),
            frame_rate=44100,  # 采样率
            sample_width=2,  # 样本宽度设置为2，表示16位样本宽度
            channels=1  # 单声道
        )
        song += note

    # 保存生成的音乐
    song.export("beautiful_music.mp3", format="mp3")

# 生成音乐
generate_music_from_array(input_array)
