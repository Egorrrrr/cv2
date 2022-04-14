#Импорт библиотек
import numpy
import sys
import time
import cv2
from numpy.lib.histograms import histogram

try:
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print('Введите путь к данным: ', end = '')
        path = str(input())
    video = cv2.VideoCapture(path) # Загрузка видео
    for attempt in range(100):
        if video.isOpened():
            break
        else:
            time.sleep(0.1)
    else:
        sys.exit('Failed to open video source')
    kernel_open = numpy.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=numpy.uint8)
    kernel_close = numpy.ones((5, 5), dtype=numpy.uint8)
    back_sub = cv2.createBackgroundSubtractorMOG2() # Использование алгоритма "Смесь Гауссиан"
    back_sub.setDetectShadows(False)
    try:
        while True:
            if cv2.waitKey(10) == 27:
                sys.exit('Stopped by user')
            success, frame = video.read()
            if not success:
                sys.exit('Stream ended')
            mask = back_sub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            cv2.imshow('Mask', mask)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, ltype=cv2.CV_16U) # Поиск связных компонентов
            for label, stat in enumerate(stats[1:], 1):
                x, y, w, h, area = stat.astype(numpy.int32)
                if area > 100:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 255), 2)
            cv2.imshow('objects', frame)
    finally:
        video.release()
except:
    print('Error')
# C:\Users\Хозяин\Downloads\bookz.mp4