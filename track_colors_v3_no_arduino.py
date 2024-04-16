import cv2
import numpy as np
import cvzone
from typing import List
# import serial

# ser = serial.Serial("/dev/ttyACM0", 9600)
cap = cv2.VideoCapture(0)
kernel = np.ones((9, 9), np.uint8)
def mask_me_open(m): return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
def mask_me_close(m): return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


class Color():
    def __init__(self, hsv: int, name="Default"):
        self.lower = (hsv-10, 100, 100)
        self.upper = (hsv+10, 255, 255)
        self.name = name

    def lower_nparray(self): return np.array(self.lower)
    def upper_nparray(self): return np.array(self.upper)


def draw(cntr, img, name=""):
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(img_stack, (x, y),
                  (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img_stack, name, (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


COLORS: List[Color] = [
    Color(110, name="azul"),
    Color(60, name="verde"),
    Color(180, name="vermelho"),
]

while (cap.isOpened()):
    ok, img = cap.read()
    img = cv2.flip(img, 1)
    assert ok, "Cannot read form device"
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_stack = cvzone.stackImages([img], 1, 1)
    for color in COLORS:
        mask = cv2.inRange(img_hsv, color.lower_nparray(),
                           color.upper_nparray())
        mask = mask_me_close(mask_me_open(mask))
        cntrs, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(cntrs) != 0:
            for cntr in cntrs:
                if cv2.contourArea(cntr) > 10:
                    draw(cntr, img_stack, color.name)
                    if color.name == "vermelho":
                        print(b'r')
                    elif color.name == "verde":
                        print(b'g')
                    elif color.name == "azul":
                        print(b'b')
    img_stack = cvzone.stackImages([img_stack, mask], 2, 1)
    cv2.imshow("Test", img_stack)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
print("Releasing: Capture device", cap.getBackendName())
cap.release()
cv2.destroyAllWindows()
