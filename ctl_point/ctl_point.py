import cv2
import numpy as np
import os
from scipy import signal
import math
import time
import glob


def morphoy_process(binary_img, kernel_close, kernel_open):
    binary_img = binary_img.astype(np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    return opening


def locate_ctl_point(img_file):
    img_root_dir = os.path.split(img_file)[0]
    img = cv2.imread(img_file)
    img, _, _ = cal_roi(img, img_root_dir)
    cv2.imwrite(os.path.join(img_root_dir, 'roi_vis.png'), img)
    r_img = img[:, :, -1]
    b_mask = img[:, :, 0] < 90
    g_mask = img[:, :, 1] < 90
    r_mask = img[:, :, 2] > 100
    red_mask = r_mask & b_mask & g_mask
    cv2.imwrite(os.path.join(img_root_dir, 'red_mask.png'), red_mask * 255)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = (gray_img < 50) * 255
    cv2.imwrite(os.path.join(img_root_dir, 'black_mask.png'), black_mask)
    kernel_close = np.ones((3, 3), np.uint8)  # 给图像闭运算定义核
    kernel_open = np.ones((3, 3), np.uint8)  # 给图像开运算定义核

    morphy_black = morphoy_process(black_mask, kernel_close, kernel_open)
    cv2.imwrite(os.path.join(img_root_dir, 'morphy_black.png'), morphy_black)

    morphy_red = morphoy_process(red_mask, kernel_close, kernel_open)
    cv2.imwrite(os.path.join(img_root_dir, 'morphy_red.png'), morphy_red)


def cal_roi(img, img_root_dir):
    h, w = img.shape[:-1]
    scale_rate = 6
    dst_h, dst_w = h // scale_rate, w // scale_rate
    img = cv2.bilateralFilter(img, 11, 50, 50)
    cv2.imwrite(os.path.join(img_root_dir, 'shuangbian.png'), img)
    r_g_mask = (img[:, :, 2].astype(np.int) - img[:, :, 1].astype(np.int)) > 50
    r_b_mask = (img[:, :, 2].astype(np.int) - img[:, :, 0].astype(np.int)) > 40
    r_mask = img[:, :, 2] > 120
    r_img = ((r_g_mask & r_b_mask & r_mask) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(img_root_dir, 'r_img.png'), r_img)

    # r_img = img[:, :, 2]
    # _, r_img = cv2.threshold(r_img, 128, 255, cv2.THRESH_BINARY)
    # img[:, :, 2] = r_img
    # cv2.imwrite(os.path.join(img_root_dir, 'red_plus.png'), img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(img_root_dir, 'gray_img.png'), gray_img)
    ada_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    cv2.imwrite(os.path.join(img_root_dir, 'ada_thresh.png'), ada_img)
    shrink_img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(img_root_dir, 'shrink_img.png'), shrink_img)
    gray_img = cv2.cvtColor(shrink_img, cv2.COLOR_BGR2GRAY)
    # black_mask = (gray_img < 30)
    # b_mask = shrink_img[:, :, 0] < 80
    # g_mask = shrink_img[:, :, 1] < 80
    r_g_mask = (shrink_img[:, :, 2].astype(np.int) - shrink_img[:, :, 1].astype(np.int)) > 50
    r_b_mask = (shrink_img[:, :, 2].astype(np.int) - shrink_img[:, :, 0].astype(np.int)) > 40
    r_mask = shrink_img[:, :, 2] > 120
    # red_mask = r_g_mask & r_b_mask
    black_coord = np.argwhere(gray_img < 50)
    red_coord = np.argwhere(r_g_mask & r_b_mask & r_mask)
    min_dist = float('inf')
    pix_pair = None
    near_length = 2
    for red_pix in red_coord:
        y_min, y_max = max(red_pix[0] - near_length, 0), min(red_pix[0] + near_length, dst_h - 1)
        x_min, x_max = max(red_pix[1] - near_length, 0), min(red_pix[1] + near_length, dst_w - 1)
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if gray_img[y, x] < 50 and x != red_pix[1] and y != red_pix[0]:
                    square_l = np.square(red_pix[0] - y) + np.square(red_pix[1] - x)
                    if min_dist > square_l:
                        min_dist = square_l
                        pix_pair = np.array([np.array([y, x]) * scale_rate, red_pix * scale_rate], dtype=np.int)
                        center_pix = (np.array([y, x]) * scale_rate + red_pix * scale_rate) // 2
        # for coord in zip(range(y_min, y_max), range(x_min, x_max)):
        #     x_near, y_near = np.abs(red_pix - black_pix) < 3
        #     if not x_near or not y_near:
        #         continue
        #     dist = np.linalg.norm(red_pix - black_pix)
        #     if dist < min_dist:
        #         min_dist = dist
        #         pix_pair = np.array([black_pix * scale_rate, red_pix * scale_rate], dtype=np.int)
        #         center_pix = (black_pix * scale_rate + red_pix * scale_rate) // 2
    print(min_dist, pix_pair)
    box_width = int(min_dist * scale_rate / 2)
    # y1, x1, y2, x2 = np.min(pix_pair[:, 0]), np.min(pix_pair[:, 1]), np.max(pix_pair[:, 0]), np.max(pix_pair[:, 1])
    y1, x1 = center_pix[0] - box_width, center_pix[1] - box_width
    y2, x2 = center_pix[0] + box_width, center_pix[1] + box_width
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.rectangle(img, (x1 - box_width, y1 - box_width), (x2 + box_width, y2 + box_width), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(img_root_dir, 'roi_vis.png'), img)
    return img, (x1 - box_width, y1 - box_width), (x2 + box_width, y2 + box_width)


def parent_contour_chk(cnt):
    area = cv2.contourArea(cnt)
    if area > 1200 or area < 50:
        return False
    perimeter = cv2.arcLength(cnt, True)
    rect = cv2.minAreaRect(cnt)

    rect_perimeter = 2 * (sum(rect[1]))
    hull_perimeter = cv2.arcLength(cv2.convexHull(cnt), True)
    thresh = area / 16
    if abs(rect[1][0] - rect[1][1]) > thresh \
            or abs(rect_perimeter - perimeter) > thresh \
            or abs(perimeter - hull_perimeter) > thresh:
        return False
    return True


def child_contour_chk(cnt, gray_img):
    # M = cv2.moments(cnt)
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    # print(cx, cy)
    mask = np.zeros(gray_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    roi = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    pixel_sum = np.sum(roi)
    pixel_count = cv2.countNonZero(roi)
    ave_pixel_val = pixel_sum / pixel_count

    # rect = cv2.boundingRect(cnt)
    #
    # rect_perimeter = 2 * (sum(rect[1]))
    # hull_perimeter = cv2.arcLength(cv2.convexHull(cnt), True)
    #
    # if abs(rect[1][0] - rect[1][1]) > 2 \
    #         or abs(rect_perimeter - perimeter) > 5 \
    #         or abs(perimeter - hull_perimeter) > 3:
    #     return False
    if ave_pixel_val > 210:
        return True
    else:
        return False


def get_edge(data, blur=False):
    if blur:
        data = cv2.GaussianBlur(data, (3, 3), 1.)
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32)
    ch_edges = []
    for k in range(data.shape[2]):
        edgex = signal.convolve2d(data[:, :, k], sobel, boundary='symm', mode='same')
        edgey = signal.convolve2d(data[:, :, k], sobel.T, boundary='symm', mode='same')
        ch_edges.append(np.sqrt(edgex ** 2 + edgey ** 2))
    return sum(ch_edges)


def get_roi_img(cnt, gray_img):
    new_cnt = np.zeros_like(cnt)
    h, w = gray_img.shape[:2]
    rect = cv2.boundingRect(cnt)
    ext = 10
    x1, y1 = rect[0] - ext if rect[0] > ext else rect[0], rect[1] - ext if rect[1] > ext else rect[1]
    x2, y2 = rect[0] + rect[2] + ext, rect[1] + rect[3] + ext
    x2 = x2 if x2 < w else w - 1
    y2 = y2 if y2 < h else h - 1
    roi_img = gray_img[y1: y2, x1: x2]
    new_cnt[:, 0, 0] = cnt[:, 0, 0] - x1 + 1
    new_cnt[:, 0, 1] = cnt[:, 0, 1] - y1 + 1
    return roi_img, new_cnt, (x1, y1)


def is_corner(gray_img, parent_cnt, child_cnt):
    # if rect[2] <

    mask_out = np.zeros(gray_img.shape[:2], np.uint8)
    cv2.drawContours(mask_out, [parent_cnt], -1, 255, -1)
    mask_in = np.ones(gray_img.shape[:2], np.uint8) * 255
    cv2.drawContours(mask_in, [child_cnt], -1, 0, -1)

    mask_ring = cv2.bitwise_and(mask_out, mask_in)
    ring_pixel_count = cv2.countNonZero(mask_ring)

    mask_in = 255 - mask_in
    roi_in = cv2.bitwise_and(gray_img, gray_img, mask=mask_in)
    cv2.imwrite('roi_in.jpg', roi_in)

    pixel_sum = np.sum(roi_in)
    if pixel_sum is None:
        print('???')
    pixel_count = cv2.countNonZero(roi_in) + 0.0001
    ave_pixel_val = pixel_sum / pixel_count

    if ave_pixel_val < 180 or not (pixel_count * 3 < ring_pixel_count < pixel_count * 6):
        return False
    return True


def contour_chk(contour, orig_gray_img):
    if not parent_contour_chk(contour):
        return None, False
    gray_img, parent_cnt, top_left = get_roi_img(contour, orig_gray_img)
    # gray_img = get_roi_img(contour, gray_img)
    if debug:
        cv2.imwrite('roi_img.png', gray_img)
    # _, thresh_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilated = cv2.m(thresh_img, kernel)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    if debug:
        cv2.imwrite('roi_thresh_img.png', thresh_img)
        # cv2.imwrite('dilated.png', dilated)
    cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)
    for i, cnt in enumerate(cnts):
        if hierarchy[i][2] == -1 and hierarchy[i][3] != -1:
            parent_idx = hierarchy[i][3]
            parent_cnt = cnts[parent_idx]
            if hierarchy[parent_idx][3] == -1:
                continue
            p_parent_cnt = cnts[hierarchy[parent_idx][3]]
            p_parent_area = cv2.contourArea(p_parent_cnt)
            parent_area = cv2.contourArea(parent_cnt)
            child_area = cv2.contourArea(cnt)
            area_rate = child_area / parent_area
            p_area_rate = parent_area / p_parent_area
            print('area rate', child_area / parent_area)
            # if is_corner(gray_img, parent_cnt, cnt):
            if 0.12 < area_rate < 0.5 and 0.3 < p_area_rate < 0.6:
                p_parent_cnt[:, 0, :] += np.array(top_left)
                parent_cnt[:, 0, :] += np.array(top_left)
                cnt[:, 0, :] += np.array(top_left)
                if debug:
                    cv2.drawContours(orig_gray_img, [p_parent_cnt, parent_cnt, cnt], -1, 0, 1)
                    cv2.imwrite('parent_cnt.png', orig_gray_img)
                return p_parent_cnt, True
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 3)
    cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)
    for i, cnt in enumerate(cnts):
        if hierarchy[i][2] == -1 and hierarchy[i][3] != -1:
            parent_idx = hierarchy[i][3]
            parent_cnt = cnts[parent_idx]
            if hierarchy[parent_idx][3] == -1:
                continue
            p_parent_cnt = cnts[hierarchy[parent_idx][3]]
            p_parent_area = cv2.contourArea(p_parent_cnt)
            parent_area = cv2.contourArea(parent_cnt)
            child_area = cv2.contourArea(cnt)
            area_rate = child_area / parent_area
            print('area rate', child_area / parent_area)
            p_area_rate = parent_area / p_parent_area
            # if is_corner(gray_img, parent_cnt, cnt):
            if 0.12 < area_rate < 0.5 and 0.3 < p_area_rate < 0.6:
                p_parent_cnt[:, 0, :] += np.array(top_left)
                parent_cnt[:, 0, :] += np.array(top_left)
                cnt[:, 0, :] += np.array(top_left)
                if debug:
                    cv2.drawContours(orig_gray_img, [p_parent_cnt, parent_cnt, cnt], -1, 0, 1)
                    cv2.imwrite('parent_cnt.png', orig_gray_img)
                return parent_cnt, True
    return None, False


def check_red_flag(img, red_points):
    red_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(red_mask, [red_points.astype(np.int32)], -1, 255, 1)
    red_img = cv2.bitwise_and(img, img, mask=red_mask)
    red_pixel_count = cv2.countNonZero(cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY))
    r_value = np.sum(red_img[:, :, 2]) / red_pixel_count
    b_value = np.sum(red_img[:, :, 0]) / red_pixel_count
    g_value = np.sum(red_img[:, :, 1]) / red_pixel_count
    min_x = np.min(red_points[:, 0])
    min_y = np.min(red_points[:, 1])
    max_x = np.max(red_points[:, 0])
    max_y = np.max(red_points[:, 1])
    red_box = img[min_y:max_y, min_x:max_x, :]
    if debug:
        cv2.imwrite('red.png', red_img)
        cv2.imwrite('redbox.png', red_box)

    red_box = red_box.reshape((-1, 3)).astype(np.int)
    rb = (red_box[:, 2] - red_box[:, 0]) > 4
    rg = (red_box[:, 2] - red_box[:, 1]) > 15
    r = red_box[:, 2] > 100
    rel = np.sum(np.logical_and(r, np.logical_and(rb, rg)))
    # if r_value - b_value > 20 and r_value - g_value > 20 and r_value > 100:
    if rel > red_box.shape[0] // 3:
        print('yes, we find red flag')
        return True
    else:
        return False


def find_code_area(img, white_box, outer_box, code_box):
    edge1 = np.linalg.norm(outer_box[0, :] - outer_box[1, :])
    edge2 = np.linalg.norm(outer_box[1, :] - outer_box[2, :])
    if edge1 > edge2:
        red_points = np.round([white_box[0, :], white_box[3, :], outer_box[3, :], outer_box[0, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[1, :], outer_box[2, :], code_box[2, :], code_box[1, :]]).astype(np.int)
            return code_points, red_points
        red_points = np.round([white_box[1, :], white_box[2, :], outer_box[2, :], outer_box[1, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[0, :], outer_box[3, :], code_box[3, :], code_box[0, :]]).astype(np.int)
            return code_points, red_points
    else:
        red_points = np.round([white_box[2, :], white_box[3, :], outer_box[3, :], outer_box[2, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[0, :], outer_box[1, :], code_box[1, :], code_box[0, :]]).astype(np.int)
            return code_points, red_points
        red_points = np.round([white_box[0, :], white_box[1, :], outer_box[1, :], outer_box[0, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[2, :], outer_box[3, :], code_box[3, :], code_box[2, :]]).astype(np.int)
            return code_points, red_points
        # return code_points, red_points
    return None, red_points


def value_2_num(value):
    r_value = value[2]
    b_value = value[0]
    g_value = value[1]
    # 黑0，白1，红2，绿3
    if g_value - b_value > 50 and g_value - r_value > 50 and g_value > 100 and b_value < 140 and r_value < 140:
        return 3
    if r_value - b_value > 50 and r_value - g_value > 50 and r_value > 100 and b_value < 140 and g_value < 140:
        return 2
    if r_value > 140 and g_value > 140 and b_value > 140:
        return 1
    if r_value < 50 and g_value < 50 and b_value < 50:
        return 0
    ave_value = np.mean(value)
    if ave_value > 140:
        return 1
    return 0


def cvt_img_2_num(img):
    if debug:
        cv2.imwrite('code_num.jpg', img)
    arr_mean = np.mean(img, axis=0)
    num_count = [0] * 4
    # 黑0，白1，红2，绿3
    for value in arr_mean:
        num = value_2_num(value)
        num_count[num] += 1
    return np.argmax(num_count)


def crop_possible_region(ctl_box, img):
    ctl_point = ctl_box[0]
    w, h = ctl_box[1]
    ave_wh = np.mean(ctl_box[1])
    angle = ctl_box[2]
    gap = ave_wh / 8 + 0.3

    white_box = cv2.boxPoints((ctl_point, (w, h), angle))

    # 宽度增加
    outer_box = cv2.boxPoints((ctl_point, (w + gap * 3, h), angle))
    code_box = cv2.boxPoints((ctl_point, (w + gap * 9, h), angle))

    code_points, red_points = find_code_area(img, white_box, outer_box, code_box)
    if code_points is None:
        # 高度增加
        outer_box = cv2.boxPoints((ctl_point, (w, h + gap * 3), angle))
        code_box = cv2.boxPoints((ctl_point, (w, h + gap * 9), angle))
        code_points, red_points = find_code_area(img, white_box, outer_box, code_box)
        if code_points is None:
            # print("Can't find red flag area")
            # cv2.drawContours(img, [code_points], -1, (255, 255, 0), 1)
            # cv2.drawContours(img, [red_points], -1, (255, 255, 0), 1)
            cv2.drawContours(img, [white_box.astype(np.int)], -1, (0, 255, 0), 1)
            cv2.circle(img, (round(ctl_point[0]), round(ctl_point[1])), 1, (0, 0, 255))
            return img, False
    # if debug:
    #     test_img = np.copy(img)
    #     cv2.drawContours(test_img, [code_points], )
    code_rect = cv2.minAreaRect(code_points)
    red_center = [np.mean(red_points[:, 0]), np.mean(red_points[:, 1])]
    code_center = [np.mean(code_points[:, 0]), np.mean(code_points[:, 1])]
    if code_center[0] == red_center[0]:
        angle = 0 if red_center[1] < code_center[1] else -180
    else:
        slope_rate = (code_center[1] - red_center[1]) / (code_center[0] - red_center[0])
        if red_center[1] < code_center[1]:
            angle = math.atan(slope_rate) / math.pi * 180
            angle = angle - 90 if angle >= 0 else 90 + angle
        elif red_center[1] > code_center[1]:
            angle = math.atan(slope_rate) / math.pi * 180
            angle = -(90 - angle) if angle < 0 else -(270 - angle)
        else:
            angle = 90 if code_center[0] < red_center[0] else -90
    # center_cord = tuple(map(round, code_center))
    w, h = max(code_rect[1]), min(code_rect[1])
    M = cv2.getRotationMatrix2D(tuple(code_center), angle, 1.0)
    # rotate_img = cv2.warpAffine(img, M, tuple(map(int, code_rect[1])), flags=cv2.WARP_INVERSE_MAP,
    rotate_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if debug:
        cv2.imwrite('rotate_img.png', rotate_img)
    warped_img = rotate_img[int(round(code_center[1] - h / 2 + 2)): int(round(code_center[1] + h / 2 - 1)),
                 int(round(code_center[0] - w / 2)): int(round(code_center[0] + w / 2)), :]
    if debug:
        cv2.imwrite('warped_code_img.png', warped_img)
    code_width = warped_img.shape[1] / 5

    x5 = warped_img[:, :round(code_width), :]
    x4 = warped_img[:, round(code_width):round(code_width * 2), :]
    x3 = warped_img[:, round(code_width * 2):round(code_width * 3), :]
    x2 = warped_img[:, round(code_width * 3):round(code_width * 4), :]
    x1 = warped_img[:, round(code_width * 4):round(code_width * 5), :]

    x5 = cvt_img_2_num(x5)
    x4 = cvt_img_2_num(x4)
    x3 = cvt_img_2_num(x3)
    x2 = cvt_img_2_num(x2)
    x1 = cvt_img_2_num(x1)

    ctl_code = x5 * pow(4, 4) + x4 * pow(4, 3) + x3 * pow(4, 2) + x2 * pow(4, 1) + x1
    # print(time.time())
    cv2.drawContours(img, [code_points], -1, (255, 255, 0), 1)
    cv2.drawContours(img, [red_points], -1, (255, 255, 0), 1)
    cv2.drawContours(img, [white_box.astype(np.int)], -1, (0, 255, 0), 1)
    cv2.circle(img, (round(ctl_point[0]), round(ctl_point[1])), 1, (0, 0, 255))
    cv2.putText(img, str(ctl_code), (int(ctl_point[0]) + 20, int(ctl_point[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (0, 0, 255), 1)
    # cv2.imwrite('possible.png', img)

    # code_rect = cv2.minAreaRect(code_points)

    # print(white_box)
    # print(outer_box)
    return img, True


def contour_find(orig_img):
    # time0 = time.time()
    # gray_img = cv2.imread(img_file, 0)
    # orig_img = cv2.imread(img_file)

    img = np.copy(orig_img)
    img[:, :, 2] = np.clip(orig_img[:, :, 2], a_min=0, a_max=200)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    max_pixel_value = np.max(gray_img)
    min_pixel_value = np.min(gray_img)
    gray_img = (orig_img[:, :, 0] * 0.11 + orig_img[:, :, 1] * 0.59 + orig_img[:, :, 2] * 0.3).astype(np.uint8)
    if debug:
        cv2.imwrite('gray_img.png', gray_img)
        # cv2.imwrite('light_img.png', light_img.astype(np.uint8))
    # gray_img = cv2.blur(gray_img, (3, 3))
    # cv2.imwrite('gray_img.jpg', gray_img)
    # dst_img = cv2.equalizeHist(gray_img)
    # cv2.imwrite('equa_hist.jpg', dst_img)
    # norm_img = np.zeros_like(gray_img)
    # cv2.normalize(gray_img, norm_img, 0, 128, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite('norm_img.jpg', norm_img)
    time1 = time.time()
    #
    # if debug:
    #     cv2.imwrite('thresh_img.png', thresh_img)
    # thresh_img = cv2.bilateralFilter(thresh_img, 5, 21, 21)
    # if debug:
    #     cv2.imwrite('MORPH_CLOSE.png', thresh_img)
    # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, (3, 3), iterations=2)
    # if debug:
    #     cv2.imwrite('MORPH_OPEN.png', thresh_img)
    _, thresh_img = cv2.threshold(gray_img, max_pixel_value * 0.9, 255, cv2.THRESH_BINARY)
    # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 3)
    if debug:
        cv2.imwrite('thresh_img.png', thresh_img)
    # 自适应阈值二值化速度太慢了，无法满足要求
    #
    # edge = get_edge(thresh_img[:, :, np.newaxis], False)
    # edge /= max(edge.max(), 0.01)
    # edge = (edge > 0.2).astype(np.uint8) * 255
    #
    # cv2.imwrite("test.png", edge)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(time.time() - time1)
    # for cnt in contours:
    hierarchy = np.squeeze(hierarchy)
    ic, parent_idx = 0, -1
    rects = []
    cnts = []
    if debug:
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        cv2.imwrite('contours0.png', img)
    for i in range(len(contours)):
        # if hierarchy[i][2] != -1 and ic == 0:
        #     parent_idx = i
        #     ic += 1
        # elif hierarchy[i][2] != -1:
        #     ic += 1
        # elif hierarchy[i][2] == -1:
        #     ic, parent_idx = 0, -1
        parent_idx = i
        if hierarchy[i][2] == -1:
            continue
        son = hierarchy[i][2]
        if hierarchy[son][2] == -1:
            continue
        grand_son = hierarchy[son][2]
        if hierarchy[grand_son][2] != -1:
            continue

        # if ic >= 2:
        # parent_idx = hierarchy[i][3]
        parent_cnt = contours[parent_idx]
        # child_cnt = contours[i]
        ic, parent_idx = 0, -1

        chk_cnt, chk_rel = contour_chk(parent_cnt, gray_img)
        if not chk_rel:
            continue
        # contour_chk(parent_cnt, gray_img)
        rect = cv2.minAreaRect(chk_cnt)

        # print('contour likely:', d1)
        rects.append(rect)
        cnts.append(parent_cnt)

    print(time.time() - time1)
    if len(rects) == 0:
        print("ctl point count:", len(rects))
        print("can't find any ctl point")
        # return
    # time1 = time.time()
    # print(time.time() - time0)
    # pts = cv2.boxPoints(rects[0]).astype(np.int)
    # img = cv2.imread(img_file)
    # cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    # img = cv2.imread(img_file)
    for rect in rects:
        orig_img, flag = crop_possible_region(rect, orig_img)
        # if flag:
        #     break
    print(time.time() - time1)
    cv2.imwrite(os.path.split(img_file)[-1], orig_img)

    print('aaaaa')


def light_tune(img_file, alpha):
    img = cv2.imread(img_file)
    tune_img = (img * (1 + alpha)).astype(np.uint8)
    # cv2.imwrite("tune_img.png", tune_img)
    return tune_img


def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    cv2.imwrite("contrast_img.png", dst)
    return dst


if __name__ == '__main__':
    # hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    # tes_t_img = r'/media/chen/wt/tmp/control_point/0929-2_D_0305_code_a3.jpg'
    # testimg = r'/media/chen/wt/tmp/control_point/0929-2_D_0826_code_a3_v2.jpg'
    # cv2.namedWindow("test", 0)
    # test_img = r'/media/chen/wt/tmp/control_point/0929-2_D_0856_2.jpg'
    test_dir = '/Users/wangtao/Desktop/work_data/smart3d/ml_tools/data/0930_err_rel/D/'
    test_imgs = glob.glob(test_dir + '/*.JPG')
    debug = True
    for img_file in test_imgs:
        # img_file = r'/Users/wangtao/Desktop/work_data/smart3d/ml_tools/data/0930_err_rel/D/test-D-0859.JPG'
        print(img_file)
        # img = light_tune(img_file, -0.3)
        # img = contrast_img(cv2.imread(img_file), 1.3, 3)
        img = cv2.imread(img_file)
        contour_find(img)
    # locate_ctl_point(test_img)
