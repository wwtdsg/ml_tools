import cv2
import os
import numpy as np
from util.img_process import correct_img


def kh_area_detect(img_file):
    root_dir = os.path.split(img_file)[0]
    src_img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img, angle = correct_img(gray_img)
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(root_dir, 'thresh_img.png'), thresh_img)
    img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, np.ones((1, 11), np.uint8))
    cv2.imwrite(os.path.join(root_dir, 'open.png'), img)
    _, contours, _ = cv2.findContours(255 - img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_area = gray_img.shape[0] * gray_img.shape[1]

    rel_cnts = []
    rel_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x - 1, y - 1, w + 2, h + 2
        cnt_area = w * h
        if cnt_area > img_area // 50 or cnt_area < img_area // 8000 \
                or w < 5 or w > 100 or h < 10 or h > 100 or w / h > 3 or h / w > 3:
            continue
        rel_areas.append(cnt_area)
        rel_cnts.append([x, y, x + w, y + h])
        # cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imwrite('/media/chen/wt/tmp/kk/boxes.png', src_img)
    return rel_cnts


def kgt_area_detect(img_file):
    root_dir = os.path.split(img_file)[0]
    src_img, angle = correct_img(cv2.imread(img_file))
    cv2.imwrite(os.path.join(root_dir, 'correct_img.png'), src_img)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((1, 2), np.uint8))
    cv2.imwrite(os.path.join(root_dir, 'MORPH_CLOSE.png'), img)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 8), np.uint8))
    cv2.imwrite(os.path.join(root_dir, 'MORPH_OPEN.png'), img)
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(root_dir, 'threshold.png'), thresh_img)
    _, contours, _ = cv2.findContours(255 - thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_area = img.shape[0] * img.shape[1]

    rel_cnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x - 2, y - 2, w + 1, h + 1
        cnt_area = w * h
        if cnt_area > img_area // 50 or cnt_area < img_area // 3500:  # 后一个阈值可以取900-2000
            continue
        rel_cnts.append([x, y, x + w, y + h])
        cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite('/media/chen/wt/tmp/kk/boxes.png', src_img)
    return rel_cnts
