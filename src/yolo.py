import torch
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

from src.metrics import yolo_iou


def pr_score(model):
    prediction_count = 0
    image_c = 0
    iou_s = []
    iou_avg = 0

    # Loop through all the samples
    for image in os.listdir("data/test"):
        # Only consider images
        if (image.endswith(".png")):
            image_c += 1
            image_code = image[:-4]

            # Get a list of iou-s for all predicted boxes
            iou = yolo_iou(image_code, model)

            prediction_count += len(iou)
            iou_s += iou

            if (len(iou) == 0):
                iou = [0]

            img_avg_iou = sum(iou)/len(iou)
            iou_avg += img_avg_iou

            # chekc if any iou value is beloow 0 or above 1 (incorrect)
            if (any(iou_single < 0 or iou_single > 1 for iou_single in iou)):
                print(f"sth wrong on {image_code}")
                exit(1)

            print(f"socre: {iou}; {image_c}/500", end="\r")

    # Loop through thresholds and save precisionn and recall
    precisions = []
    recalls = []
    for threshold in np.arange(0, 1.1, 0.1):
        # For each threshold TP
        TP = 0

        for iou in iou_s:
            # If iou is above threshold this is TP
            if (iou > threshold):
                TP += 1

        # TP / all predictions (boxes)
        precision = TP / prediction_count
        # TP / all ears (every image contains one ear)
        recall = TP / image_c

        precisions.append(precision)
        recalls.append(recall)

    with open("results/yolo_precisionn_recall.txt", "w") as pr:
        pr.write(str(precisions) + "\n")
        pr.write(str(recalls))

    print()

    with open("results/jv_res.txt", "a") as f:
        # Devide by number of samples
        iou_avg /= image_c

        # Average precision and recall over all
        avg_recall = sum(recalls)/len(recalls)
        avg_precision = sum(precisions)/len(precisions)

        f.write(f"yolo => AvgIoU: {iou_avg}; avg_recall: {avg_recall}; avg_precision: {avg_precision}\n") # noqa
    print(f"yolo => AvgIoU: {iou_avg}; avg_recall: {avg_recall}; avg_precision: {avg_precision}\n") # noqa

    plt.plot(recalls, precisions)
    plt.savefig("results/yolo.png")


def mark_image(image_code):
    image = cv.imread(f"data/test/{image_code}.png")
    img_height = image.shape[0]
    img_width = image.shape[1]

    with open(f"data/test/{image_code}.txt") as f:
        line = f.read().split()
        box_width = round(float(line[3])*img_width)
        box_height = round(float(line[4])*img_height)
        box_x = round(float(line[1])*img_width-(box_width/2))
        box_y = round(float(line[2])*img_height-(box_height/2))

    model = torch.hub.load('./yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")

    img_path = f"data/test/{image_code}.png"

    results = model(img_path).xyxy[0].numpy()

    for box in results:
        image = cv.rectangle(image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (255, 0, 0), 2)

    image = cv.rectangle(image, (box_x, box_y+box_height), (box_x+box_width, box_y), (0, 0, 255), 2)

    cv.imwrite(f"{image_code}_marked_yolo.png", image)
