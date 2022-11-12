from src.jv import pr_score
import cv2 as cv
import time

minNeighbours_list = [1, 2, 3, 4, 5, 8, 10]
scaleFactor_list = [1.05, 1.1, 1.2, 1.3, 1.6, 2.0]

minNeighbours_list = [10]
scaleFactor_list = [2.0]
"""
minNeighbours_list = [3]
scaleFactor_list = [1.1]"""

right_cascade = cv.CascadeClassifier()
left_cascade = cv.CascadeClassifier()

right_cascade.load("haarcascade_mcs_rightear.xml")
left_cascade.load("haarcascade_mcs_leftear.xml")

for minNeighbours in minNeighbours_list:
    for scaleFactor in scaleFactor_list:
        start = time.time()
        pr_score(right_cascade, left_cascade, minNeighbours, scaleFactor)
        exec_time = time.time()-start
        print(f"Execution time: {exec_time}")
