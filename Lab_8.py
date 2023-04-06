# Вариант 5
import numpy as np
import random
import cv2

# Задание 1
def noise(image, prob):
    output = np.zeros(image.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread('variant-5.jpg', cv2.IMREAD_GRAYSCALE)
noise_img = noise(image, 0.05)
cv2.imwrite('noise.jpg', noise_img)

# Задание 2 + Доп.задание
def overlay(background, img, x, y): # Вырезаем муху и помещаем её на место
    b = np.copy(background)
    place = b[y: y + img.shape[0], x: x + img.shape[1]]
    a = img[..., 3:].repeat(3, axis=2).astype('uint16')
    place[...] = (place.astype('uint16') * (255 - a) // 255) + img[..., :3].astype('uint16') * a // 255
    return b

rplce = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
cap = cv2.VideoCapture(0)
down_points = (640, 480)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)  # Сжатие картинки
    secure = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)  # Поиск чёрного на белом
    contours, hierarchy = cv2.findContours(thresh,
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:  # Если нашли
        c = max(contours, key=cv2.contourArea)  # Берём наибольший контур
        x, y, w, h = cv2.boundingRect(c)  # Находим величины этого контура
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        contours = cv2.drawContours(frame, [box], 0, (0, 0, 0), 2)
        center = ((int((x + (w // 2) - 32))), int((y + (h // 2) - 32)))  # Координаты центра
        x_center = center[0]
        y_center = center[1]
        if center[0] < 120 and center[1] < 120:  # Смена цвета контура при попадании в определённые участки
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
        elif center[0] > 480 and center[1] > 360:
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        else:
            cv2.drawContours(frame, [box], 0, (0, 0, 0), 2)
        if not (64 < x_center < 640 - 64):
            if x_center < 64:
                x_center = 64
            else:
                x_center = 640 - 64
        if not (64 < y_center < 480 - 64):
            if y_center < 64:
                y_center = 64
            else:
                y_center = 480 - 64
        cv2.imshow('replace', overlay(frame, rplce, x_center, y_center))
    else:
        cv2.imshow('replace', secure)  # Подстраховка при отсутствии объекта
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
