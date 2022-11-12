import cv2 as cv


def area_of_intersection(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2):
    if (min(maxx1, maxx2) < max(minx1, minx2) or min(maxy1, maxy2) < max(miny1, miny2)):
        return 0

    w = min(maxx1, maxx2) - max(minx1, minx2)
    h = min(maxy1, maxy2) - max(miny1, miny2)

    return w*h


def area_of_union(width1, height1, width2, height2, ai):
    return (width1*height1)+(width2*height2)-ai


def _iou(box1_x, box1_y, box1_width, box1_height, box2_x, box2_y, box2_width, box2_height):
    ai = area_of_intersection(box1_x, box1_y, box1_x+box1_width, box1_y+box1_height, box2_x, box2_y, box2_x+box2_width,
                              box2_y+box2_height)

    au = area_of_union(box1_width, box1_height, box2_width, box2_height, ai)

    return ai/au


def iou(boxes, true_x, true_y, true_width, true_height):
    r = []

    for box in boxes:
        r.append(_iou(true_x, true_y, true_width, true_height, box[0], box[1], box[2], box[3]))

    return r


def jv_iou(image_code, right_cascade, left_cascade, minNeighbours=3, scaleFactor=1.1):
    image = cv.imread(f"data/test/{image_code}.png", cv.IMREAD_GRAYSCALE)
    img_height = image.shape[0]
    img_width = image.shape[1]

    with open(f"data/test/{image_code}.txt") as f:
        line = f.read().split()
        box_width = round(float(line[3])*img_width)
        box_height = round(float(line[4])*img_height)
        box_x = round(float(line[1])*img_width-(box_width/2))
        box_y = round(float(line[2])*img_height-(box_height/2))

    rights = right_cascade.detectMultiScale(image, minNeighbors=minNeighbours, scaleFactor=scaleFactor)
    lefts = left_cascade.detectMultiScale(image, minNeighbors=minNeighbours, scaleFactor=scaleFactor)

    boxes = []
    if (len(rights) > 0):
        boxes += list(rights)
    if (len(lefts) > 0):
        boxes += list(lefts)

    r_s = iou(boxes, box_x, box_y, box_width, box_height)

    return r_s


def iou_for_yolo(boxes, true_x, true_y, true_width, true_height):
    r = []

    for box in boxes:
        # print([true_x, true_y, true_width, true_height])
        # print(box)
        r.append(_iou(true_x, true_y, true_width, true_height, box[0], box[1], box[2]-box[0], box[3]-box[1]))

    return r


def yolo_iou(image_code, model):
    image = cv.imread(f"data/test/{image_code}.png")
    img_height = image.shape[0]
    img_width = image.shape[1]

    with open(f"data/test/{image_code}.txt") as f:
        line = f.read().split()
        box_width = round(float(line[3])*img_width)
        box_height = round(float(line[4])*img_height)
        box_x = round(float(line[1])*img_width-(box_width/2))
        box_y = round(float(line[2])*img_height-(box_height/2))

    results = model(f"data/test/{image_code}.png").xyxy[0].numpy()

    r_s = iou_for_yolo(results, box_x, box_y, box_width, box_height)

    return r_s


def precision_recall_graph():
    pass
