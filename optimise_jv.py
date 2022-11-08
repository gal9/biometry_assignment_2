from src.jv import pr_score
import cv2 as cv

minNeighbours_list = [1, 2, 3, 4, 5, 8, 10]
scaleFactor_list = [1.0, 1.1, 1.2, 1.3, 1.6, 1.9, 2.0]

right_cascade = cv.CascadeClassifier()
left_cascade = cv.CascadeClassifier()

right_cascade.load("haarcascade_mcs_rightear.xml")
left_cascade.load("haarcascade_mcs_leftear.xml")

for minNeighbours in minNeighbours_list:
    for scaleFactor in scaleFactor_list:
        pr_score(right_cascade, left_cascade, minNeighbours, scaleFactor)
