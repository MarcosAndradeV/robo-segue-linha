import cv2
import numpy as np
import cvzone
from typing import List
import simple_pid
# import serial

# ser = serial.Serial("/dev/ttyACM0", 9600)
cap = cv2.VideoCapture(0)
kernel = np.ones((9, 9), np.uint8)
def mask_me_open(m): return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
def mask_me_close(m): return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


class Color():
    def __init__(self, hsv: int, name="Default"):
        self.lower = (hsv-10, 100, 30)
        self.upper = (hsv+10, 255, 255)
        self.name = name

    def lower_nparray(self): return np.array(self.lower)
    def upper_nparray(self): return np.array(self.upper)


def draw(cntr, img, name=""):
    x, y, w, h = cv2.boundingRect(cntr)
    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)


COLORS: List[Color] = [
    Color(110, name="azul"),
    Color(60, name="verde"),
    Color(180, name="vermelho"),
    Color(20, name="preto")
]

while (cap.isOpened()):
    ok, img = cap.read()
    img = cv2.flip(img, 1)
    assert ok, "Cannot read form device"
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_stack = cvzone.stackImages([img], 1, 1)
    color = COLORS[-1]
    mask = cv2.inRange(img_hsv, np.array([0, 0, 0]),
                       np.array([180, 255, 30]))
    mask = mask_me_close(mask_me_open(mask))
    cntrs, hy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if (cont_cntr := len(cntrs)) != 0:
        print("CONTORNOS: ", cont_cntr)
        draw(cntrs[0], img_stack)
        # for cntr in cntrs:
        # print(hy[0])
        # x, y, w, h = cv2.boundingRect(cntr)
        # print("x = ", x)
        # print("w = ", w)
        # print("h = ", h)
        # print("y = ", y)
    img_stack = cvzone.stackImages([img_stack, mask], 2, 1)
    cv2.imshow("Test", img_stack)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
print("Releasing: Capture device", cap.getBackendName())
cap.release()
cv2.destroyAllWindows()
