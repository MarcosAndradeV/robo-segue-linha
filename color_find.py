import cv2
import numpy as np


red = np.array([255, 0, 0])

cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
