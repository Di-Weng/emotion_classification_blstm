# -*- coding: utf-8 -*-
"""
-------------------------------
   Time    : 2018-12-04 14:28
   Author  : diw
   Email   : di.W@hotmail.com
   File    : get_audio.py
   Desc:
-------------------------------
"""
import pyaudio
import wave
input_filename = "input.wav"
input_filepath = ""
input_path = input_filepath + input_filename

def microphone_audio(filepath):
    aa = str(input("是否开始录音？   （y/n）"))
    if aa == str("y") :
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 2
        WAVE_OUTPUT_FILENAME = filepath
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*"*10, "开始录音：请在2秒内输入语音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*"*10, "录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif aa == str("n"):
        return

if __name__=='__main__':
    microphone_audio(input_path)