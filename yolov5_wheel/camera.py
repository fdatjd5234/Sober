#!/user/bin/python3
# -*- coding: UTF-8 -*-
"""
@author:quyang
@file:camera.py
@time:2023/02/17-16:17
"""


"""
原来的代码----曲洋
import time

import cv2


if __name__ == '__main__':

    url = "rtsp://admin:cp6_ca123@10.200.35.250/Streaming/Channels/2"
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    index = 1
    while ret:
        ret, frame = cap.read()

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    """
import cv2
import time

# 摄像头URL列表
urls = [
    "rtsp://admin:cp6_ca123@10.200.35.250/Streaming/Channels/2",  # 第一个摄像头的URL
    "rtsp://admin:cp6_ca123@10.200.35.24/Streaming/Channels/2",  # 第二个摄像头的URL
]

# 创建摄像头对象列表
caps = [cv2.VideoCapture(url) for url in urls]

while True:
    frames = []
    
    # 读取每个摄像头的一帧图像
    for cap in caps:
        ret, frame = cap.read()
        frames.append(frame)
    
    # 显示图像
    for i, frame in enumerate(frames):
        cv2.imshow(f"Camera {i+1}", frame)
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头对象和关闭窗口
for cap in caps:
    cap.release()
cv2.destroyAllWindows()

