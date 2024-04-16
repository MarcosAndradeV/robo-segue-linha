import cv2
import numpy as np

class LineTracking():
    """
    Classe permettant le traitement d'image, la délimitation d'un contour et permet de trouver le centre de la
    forme detectée
    """
    def __init__(self):
        """The constructor."""
        self.cap = cv2.VideoCapture(2)  # Open the camera (0 is the default camera)
        self.img_inter = None
        self.img_final = None
        self.centroids = []
        self.mean_centroids = [0, 0]

    def processing(self):
        """Méthode permettant le traitement d'image"""
        ret, frame = self.cap.read()  # Capture a frame from the camera
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)  # Binarize the image

        self.img_inter = thresh

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        connectivity = 8
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        self.centroids = output[3]

        for c in self.centroids:
            self.mean_centroids[0] += c[0] / len(self.centroids)
            self.mean_centroids[1] += c[1] / len(self.centroids)

        self.img_final = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for c in self.centroids:
            self.img_final[int(c[1]) - 5: int(c[1]) + 10, int(c[0]) - 5: int(c[0]) + 10] = [0, 255, 0]

if __name__ == '__main__':
    test = LineTracking()
    while True:
        test.processing()
        cv2.imshow('camera', test.img_final)
        print(test.centroids)
        key = cv2.waitKey(1)
        if key == ord(' '):
            break
    cv2.destroyAllWindows()
