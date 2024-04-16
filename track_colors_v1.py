import cv2
import numpy as np
import cvzone

"""
hsv -> sistemas de cores formadas por Hue (matriz), saturação e value
    hue (tonalidade) -> tipo de cor, abrangendo um espectro
    saturação -> quanto menor o valor, mais próximo de cinza será a img
    valor -> define o brilho
"""
# Define um range para a busca de uma tonalidade
# lower = np.array([28, 50, 50])  # AMARELO
# upper = np.array([33, 255, 255])  # AMARELO
# lower = np.array([110, 70, 70])  # AZUL
# upper = np.array([120, 255, 255])  # AZUL

lower = {
    "AMARELO": np.array([28, 70, 70]),
    "AZUL": np.array([110, 70, 70]),
    "BLACK": np.array([0, 0, 0])
}

upper = {
    "AMARELO": np.array([33, 255, 255]),
    "AZUL": np.array([120, 255, 255]),
    "BLACK": np.array([0, 0, 0])
}


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # converte para hsv
    imgStack = cvzone.stackImages([img], 1, 1)

    for key, _ in lower.items():
        mask = cv2.inRange(image_hsv, lower[key], upper[key])
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contornos) != 0:
            # print(key)
            for contorno in contornos:
                if cv2.contourArea(contorno) > 100:
                    # print(cv2.contourArea(contorno))
                    x, y, w, h = cv2.boundingRect(contorno)
                    cv2.rectangle(imgStack, (x, y),
                                  (x + w, y + h), (0, 0, 255), 2)
                    cx = x + w // 2
                    cy = y + h // 2
                    cv2.putText(imgStack, key + " object", (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    cv2.circle(imgStack, (cx, cy), 2, (0, 0, 255), -1)
    imgStack = cvzone.stackImages([imgStack, mask], 2, 1)
    cv2.imshow("Image", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
