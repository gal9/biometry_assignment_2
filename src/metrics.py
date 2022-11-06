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


def iou(lefts, rights, true_x, true_y, true_width, true_height):
    r = []

    for right in rights:
        r.append(_iou(true_x, true_y, true_width, true_height, right[0], right[1], right[2], right[3]))

    for left in lefts:
        r.append(_iou(true_x, true_y, true_width, true_height, left[0], left[1], left[2], left[3]))

    return r


def jv_iou(image_code, right_cascade, left_cascade, minNeighbours=3, scaleFactor=1.1):
    image = cv.imread(f"data/test/{image_code}.png")
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

    r_s = iou(lefts, rights, box_x, box_y, box_width, box_height)

    return r_s


def precision_recall_graph():
    pass
