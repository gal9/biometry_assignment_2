from src.metrics import jv_iou
import os
import cv2 as cv
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt


def pr_score(right_cascade, left_cascade, minNeighbours=3, scaleFactor=1.1):
    c = 0
    image_c = 0
    iou_s = []
    iou_avg = 0

    for image in os.listdir("data/test"):
        if (image.endswith(".png")):
            image_c += 1
            image_code = image[:-4]
            iou = jv_iou(image_code, right_cascade, left_cascade, minNeighbours, scaleFactor)
            iou_s.append(iou)
            c += len(iou)
            iou_avg += sum(iou)

            # chekc if any iou value is beloow 0 or above 1 (incorrect)
            if (any(iou_single < 0 or iou_single > 1 for iou_single in iou)):
                print(f"sth wrong on {image_code}")
                exit(1)

            print(f"socre: {iou}; {image_c}/500", end="\r")

    # Loop through thresholds and save precisionn and recall
    precisionns = []
    recalls = []
    for threshold in np.arange(0, 1.1, 0.1):
        # For each threshold count metrics
        TP = 0
        FP = 0
        FN = 0

        for image_ious in iou_s:
            # Check if ear is found
            ear_found = False

            # Loop through ious of specific image
            for iou in image_ious:
                # If iou is above threshold this is TP
                if (iou > threshold):
                    TP += 1
                    ear_found = True

                # If it is below this is FP (predicted a box in incorect place)
                else:
                    FP += 1

            # If no ear was found in the picture this is FN (assuming every picture has exacly one ear)
            # (did not predict a box where it should)
            if (not ear_found):
                FN += 1

        precisionn = TP / (TP + FP)
        recall = TP / (TP + FN)

        precisionns.append(precisionn)
        recalls.append(recall)

    with open(f"results/MN{minNeighbours}_SF{scaleFactor}_precisionn_recall.txt", "w") as pr:
        pr.write(str(precisionns) + "\n")
        pr.write(str(recalls))

    print()

    with open("results/jv_res.txt", "a") as f:
        iou_avg /= c
        AUC = metrics.auc(recalls, precisionns)

        f.write(f"minNeighbours: {minNeighbours}; scaleFactor: {scaleFactor} => AvgIoU: {iou_avg}; AUC: {AUC}\n")
    print(f"minNeighbours: {minNeighbours}; scaleFactor: {scaleFactor} => AvgIoU: {iou_avg}; AUC: {AUC}\n")

    plt.plot(recalls, precisionns)
    plt.savefig(f"results/MN{minNeighbours}:SF{scaleFactor}.png")


def mark_image(image_code, minNeighbours, scaleFactor):
    image = cv.imread(f"data/test/{image_code}.png")
    img_height = image.shape[0]
    img_width = image.shape[1]

    with open(f"data/test/{image_code}.txt") as f:
        line = f.read().split()
        box_width = round(float(line[3])*img_width)
        box_height = round(float(line[4])*img_height)
        box_x = round(float(line[1])*img_width-(box_width/2))
        box_y = round(float(line[2])*img_height-(box_height/2))

    right_cascade = cv.CascadeClassifier()
    left_cascade = cv.CascadeClassifier()

    right_cascade.load("haarcascade_mcs_rightear.xml")
    left_cascade.load("haarcascade_mcs_leftear.xml")

    rights = right_cascade.detectMultiScale(image, minNeighbours=minNeighbours, scaleFactor=scaleFactor)
    lefts = left_cascade.detectMultiScale(image, minNeighbours=minNeighbours, scaleFactor=scaleFactor)

    for left in lefts:
        image = cv.rectangle(image, (left[0], left[1]+left[3]), (left[0]+left[2], left[1]), (255, 0, 0), 2)

    for right in rights:
        image = cv.rectangle(image, (right[0], right[1]+right[3]), (right[0]+right[2], right[1]), (0, 255, 0), 2)

    image = cv.rectangle(image, (box_x, box_y+box_height), (box_x+box_width, box_y), (0, 0, 255), 2)

    cv.imwrite(f"{image_code}_marked.png", image)
