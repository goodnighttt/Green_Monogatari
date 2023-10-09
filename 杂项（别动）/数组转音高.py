import numpy as np
from pydub import AudioSegment

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

# 生成随机数据数组，假设这是你的输入数据
random_data = np.random.randint(0, 7, size=10)

# 定义音符持续时间（以毫秒为单位）
note_duration = 1000

# 创建一个空的音频段
music = AudioSegment.empty()

# 将随机数据映射到音高频率并添加到音乐中
for note_index in random_data:
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

# # 播放音乐（如果你想听到它）
# music.export("output_music.wav", format="wav").play()
