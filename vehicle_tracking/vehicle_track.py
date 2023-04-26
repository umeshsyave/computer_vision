import cv2
from tracker import *

# elucidatedisttracker to store the detections in dict with particular id
objects_count = EuclideanDistTracker()

obj_detect = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=90)

video = cv2.VideoCapture('vehicle_track/traffic.mp4')
count = set()

while True:
    ret, frame = video.read()
    region = frame[100:500, 200:800]  # selecting the region to detect
    bin_fr = obj_detect.apply(region)
    _, bin_fr = cv2.threshold(bin_fr, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        bin_fr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # filtering out small unneccesary detections
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(region, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detections.append([x, y, w, h])

    bboxids = objects_count.update(detections)

    for id in bboxids:
        cv2.putText(region, str(id[4]), (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        count.add(id[4])

    cv2.imshow('monitoring', frame)
    # cv2.imshow('binary_visual',bin_fr)

    if cv2.waitKey(1) & 0XFF == 27:
        break

print(len(count))

video.release()
cv2.destroyAllWindows()
