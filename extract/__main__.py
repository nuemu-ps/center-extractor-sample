import time
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter('output.mp4', fmt, fps, (width, height), True)

def contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(binary, 0, 250)
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    mask = np.zeros_like(img)
    mask = cv2.drawContours(mask, max_contour, -1, 255, 1)
    (y, x, z) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    output = (topy, bottomy, topx, bottomx)

    return output

def find(img):
    (topy, bottomy, topx, bottomx) = contours(img)
    crop = img[topy:bottomy+1, topx:bottomx+1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=60, minRadius=0, maxRadius=200)
    if len(circles[0, :]) <= 7:
        for circle in circles[0, :]:
            img = cv2.circle(img, (topx + int(circle[0]), topy + int(circle[1])), 3, (255, 0, 0), 3)

    return img


while True:
    ret, img = cap.read()
    if ret:
        img = find(img)
        cv2.imshow('video', img)
        writer.write(img)
        time.sleep(0.1)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
writer.release()
cv2.destroyAllWindows()
