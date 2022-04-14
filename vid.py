# Импорт библиотек
import numpy
import sys
import time
import cv2
import time
from numpy.lib.histograms import histogram


def validate(old_rect, new_rect):
    if old_rect[0][0] == 0 and old_rect[0][1] == 0:
        return True
    centre_old = ((old_rect[0][0] + old_rect[1][0]) / 2, (old_rect[0][1] + old_rect[1][1]) / 2)
    centre_new = ((new_rect[0][0] + new_rect[1][0]) / 2, (new_rect[0][1] + new_rect[1][1]) / 2)
    if abs(centre_new[0] - centre_old[0]) < 40 and abs(centre_new[1] - centre_old[1]) < 40:
        return True
    else:
        return False


try:
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print('Введите путь к данным: ', end='')
        path = "vid2.mp4"
    video = cv2.VideoCapture(path)  # Загрузка видео
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
    back_sub = cv2.createBackgroundSubtractorMOG2()  # Использование алгоритма "Смесь Гауссиан"
    back_sub.setDetectShadows(False)
    rect = None
    old_time = None
    old_dis = None
    kmh = 0
    try:
        while True:
            if cv2.waitKey(10) == 27:
                sys.exit('Stopped by user')
            success, frame = video.read()
            if not success:
                sys.exit('Stream ended')
            mask = back_sub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_RECT, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close)
            cv2.imshow('Mask', mask)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,
                                                                                ltype=cv2.CV_32S)  # Поиск связных компонентов
            for label, stat in enumerate(stats[1:], 1):
                x, y, w, h, area = stat.astype(numpy.int32)
                if area > 200:
                    if rect is not None:
                        if validate(rect, [(x, y), (x + w, y + h)]):
                            if old_time is None:
                                old_time = time.perf_counter()
                            else:
                                time_two = time.perf_counter()
                                if time_two - old_time > 2:
                                    old_time = time_two
                                    if old_dis is None:
                                        old_dis = (x + x + w) / 2
                                    else:
                                        meters = ((x + x + w) / 2 - old_dis) / 44
                                        kmh = meters/2 * 3.6
                                        print(round(kmh, 2))
                                        old_dis = (x + x + w) / 2
                            rect = [(x, y), (x + w, y + h)]
                            cv2.rectangle(frame, ((x + x + w) // 2, y), ((x + x + w+10) // 2, (y + h)), (255, 0, 0), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 255, 255), 2)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (720, 50)
                            fontScale = 1
                            color = (0, 0, 0)
                            thickness = 2
                            image = cv2.putText(frame,str(round(kmh, 2)) , org, font,
                                                fontScale, color, thickness, cv2.LINE_AA)


                    else:
                        rect = [(x, y), (x + w, y + h)]
            cv2.imshow('objects', frame)
    finally:
        video.release()
except Exception as e:
    print(e)
