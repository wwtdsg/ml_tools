import cv2
import base64
import numpy as np
import string
import os
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import operator
from urllib import parse
import re
import glob
from skimage import io
import json


def is_base64(input_string):
    """
    检查输入字符串是否为base64编码
    :param input_string:
    """
    if not input_string:  # 空
        return False

    if not isinstance(input_string, str):  # 确保类型
        return False

    if len(input_string) % 4 != 0:  # 能被4整除
        return False
    # base64的字符必定属于[0-9,a-z,A-Z,+,/,=]
    b64_char_set = string.ascii_letters + string.digits + "+/="

    b64_flag = True
    for input_char in input_string:
        if input_char not in b64_char_set:
            b64_flag = False
            break

    return b64_flag


def color2gray(img_path, save_path):
    img = cv2.imread(img_path, 0)
    cv2.imwrite(save_path, img)


def decode_image_with_base64(im_str, _type=cv2.IMREAD_COLOR):
    """
    从base64字符串解码为numpy.array
    :param base64_str: base64字符串
    :return:
        code: 返回码, 0表正常,非0表异常
        color_img: 彩色图片,numpy.array格式
    """
    code = 0

    b64_flag = is_base64(im_str)  # 判断输入字符串合法性
    if not b64_flag:
        code, message = 4, "fail"
        return code, message

    im_byte = base64.b64decode(im_str)
    im_arr = np.fromstring(im_byte, np.uint8)
    color_img = cv2.imdecode(im_arr, _type)
    # color_img = cv2.imdecode(im_arr, cv2.IMREAD_GRAYSCALE)

    if color_img is None:
        code, message = 5, "fail"
        return code, message

    return code, color_img


def visualize(im_name, im_save_name, boxes_coord, labellist, font, subject=None, vis_image=True):
    """
    检测与识别结果可视化.
    :param im_name: 图片路径/图片url
    :param boxes_coord: 检测框位置信息[[xmin, ymin, xmax, ymax],...]
    :param labellist: 检测框识别结果字符串[res_string1, res_string2]
    :param out_dir: 结果写出目录
    :param font: 字体类型及大小
    :param subject: 科目类别
    """
    print('#############')
    if not boxes_coord:  # 保证检测框个数>=1
        return
    # boxes_coord = np.asarray(boxes_coord, dtype=int)/scale  # 缩放到原图的比例坐标
    # boxes_coord = boxes_coord.astype(int)
    # boxes_coord = boxes_coord.tolist()

    try:
        im = io.imread(im_name) if vis_image else None  # RGB
    except Exception as e:
        print(e)
        return

    # im, angle = correct_image(im)
    im = Image.fromarray(im) if vis_image else None

    im_w, im_h = im.size[:2] if vis_image else (None, None)

    y_centers = {}
    for count, coord in enumerate(boxes_coord):
        y_centers[count] = (coord[1] + coord[3]) // 2  # # ymin = coord[1], ymax = coord[3]

    # 按检测框的中心y点排序
    y_centers = sorted(y_centers.items(), key=operator.itemgetter(1))

    lines = []  # 总共的行集合列表
    line = []  # 高度接近的检测框集合列表
    line.append(y_centers[0])  # 初始化第一行
    for y_center in y_centers[1:]:
        idx, y_c = y_center  # idx表示第几个检测框,y_c表示该检测框的y轴中心点
        if (y_c - line[-1][1]) < ((boxes_coord[idx][3] - boxes_coord[idx][1]) / 2):
            line.append(y_center)
        else:
            lines.append(line)
            line = []
            line.append(y_center)
    lines.append(line)

    # 判断大概需要的图像宽度和高度
    new_image_w = 0
    new_image_h = 0
    for line in lines:
        if len(line) == 1:
            idx = line[0][0]  # 第idx个检测框
            label_w, label_h = font.getsize(labellist[idx])  # 返回的是w,h

        else:
            linelabel = ""
            # for index, l in enumerate(line):
            for index in range(len(line)):
                linelabel = linelabel + labellist[index] + '   '
            label_w, label_h = font.getsize(linelabel)  # 返回的是w,h
            # print(linelabel, label_w)
        if vis_image:
            new_image_w = max(new_image_w, label_w)
            new_image_h += label_h

            new_image_h = max(new_image_h, im_h)

            # newimg = Image.new('RGB', (int(im_w * 2.3), int(im_h * 1.1)), (255, 255, 255))
            newimg = Image.new("RGB", (new_image_w + im_w + 10, new_image_h + 10), (255, 255, 255))
            newimg.paste(im, (0, 0))
            txt_draw = ImageDraw.Draw(newimg)

            # print('vising', out_dir)

    # 方式1
    # base_name = "_".join(im_name.split("/")[-2:])
    # 方式2
    # base_name = os.path.basename(im_name)
    # subject = im_name.split("/")[-2]
    # out_dir = os.path.join(out_dir, subject)
    # 方式3
    if im_name.endswith(("png", "PNG", "jpg", "JPG", "JPEG", "jpeg")):
        basename = "_".join(im_name.split("/")[-1:])  # url
        # basename = os.path.basename(im_name)    # im_name
    else:
        raise IOError

    base_name = basename
    # base_name = "_".join(im_name.split("/")[-3:])
    stem, ext = os.path.splitext(base_name)
    txt_basename = stem + "_gz_text.txt"
    # if  path_perfect:
    #     txt_basename= os.path.basename(txt_basename)
    #     base_name = os.path.basename(base_name)
    txt_basename = os.path.basename(txt_basename)
    base_name = os.path.basename(base_name)

    if vis_image:
        draw_start_x = im_w + 10  # 绘图的起始x
        draw_start_y = 10  # 绘图的起始y
        curr_x, curr_y = draw_start_x, draw_start_y

    min_color = int(255 / (len(boxes_coord) + 1))
    for count, line in enumerate(lines):
        if len(line) == 1:
            idx = line[0][0]  # 第idx个检测框
            xmin, ymin, xmax, ymax = boxes_coord[idx]
            if vis_image:
                # txt_draw.rectangle((xmin, ymin, xmax, ymax), outline=(0,255,0), width=2)
                txt_draw.rectangle((xmin, ymin, xmax, ymax), outline=(0, 255, min_color * idx))
                txt_draw.text((draw_start_x, curr_y), labellist[idx], fill=6, font=font)
                label_w, label_h = font.getsize(labellist[idx])  # 返回的是w,h
                curr_x += label_w
                curr_y += label_h
        else:
            orilineboxes = {}  # 同一个line的检测框按横坐标xmin集合
            linelabels = []
            for idx, line_item in enumerate(line):  # line_item: 一行的每个检测框xmin
                orilineboxes[idx] = boxes_coord[line_item[0]]
                linelabels.append(labellist[line_item[0]])

            # 一行的检测框集合按xmin排序
            orilineboxes = sorted(orilineboxes.items(), key=operator.itemgetter(1))
            linelabel = ''
            for orilinebox in orilineboxes:
                idx, coord = orilinebox
                xmin, ymin, xmax, ymax = coord
                if vis_image:
                    txt_draw.rectangle((xmin, ymin, xmax, ymax), outline=(0, 255, min_color * idx))
                linelabel = linelabel + linelabels[idx] + '   '
            if vis_image:
                txt_draw.text((draw_start_x, curr_y), linelabel, fill=6, font=font)
                label_w, label_h = font.getsize(linelabel)  # 返回的是w,h
                curr_x += label_w
                curr_y += label_h
    if vis_image:
        newimg.save(im_save_name)


def img2base64(im_path):
    """
    将图片转换为对应的base64编码.
    :param im_path: 图片路径
    :return: 图片的base64编码
    """
    with open(im_path, "rb") as frb:  # 获取图片二进制数据
        im_byte = frb.read()

    im_base64 = base64.b64encode(im_byte)
    im_base64 = str(im_base64, "utf-8")  # bytes -> base64 string

    return im_base64


def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);


def warp_rect_img(rect, src_img):
    width = int(rect[1][0])
    height = int(rect[1][1])
    box = cv2.boxPoints(rect)
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(src_img, M, (width, height))
    return warped_img


def find_rect_corner(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dst_1 = (cv2.goodFeaturesToTrack(img_gray, 45, 0.01, 5)).squeeze().astype(np.int32)
    # img_gray[dst_1[:, 1], dst_1[:, 0]] = 255
    #
    # cv2.imwrite('goodFeaturesToTrack.png', img_gray)

    dst = cv2.cornerHarris(np.float32(img_gray), 2, 5, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, cv2.THRESH_BINARY)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners: (type, maxCount, epsilon）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    # Refines the corner locations
    corners = cv2.cornerSubPix(img_gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # Now draw them
    # res = np.hstack((centroids, corners))
    # np.int0 可以用来省略小数点后面的数字
    res = np.int0(corners)
    res = res[np.where(np.logical_and(res[:, 1] > img.shape[0] * 0.4, res[:, 1] < img.shape[0] * 0.6))]
    mean_height = np.mean(res[:, 1])

    res = sorted(list(res), key=lambda item: item[0])
    width_lists = np.array([res[i][0] - res[i - 1][0] for i in range(1, len(res)) if res[i][0] - res[i - 1][0] != 0])
    var_dis = np.var(width_lists)

    for i in range(len(res) // 2):
        if var_dis > 1:
            max_index = np.argmax(width_lists)
            width_lists = np.concatenate((width_lists[:max_index], width_lists[max_index + 1:]), axis=0)
            var_dis = np.var(width_lists)
        else:
            break
        if var_dis > 1:
            min_index = np.argmin(width_lists)
            width_lists = np.concatenate((width_lists[:min_index], width_lists[min_index + 1:]), axis=0)
            var_dis = np.var(width_lists)
        else:
            break
    res = np.array(res)
    img[res[:, 1], res[:, 0]] = [0, 255, 0]

    cv2.imwrite('subpixel.png', img)

    corner_img = np.copy(img)
    corner_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite('corner.png', corner_img)
    cv2.imwrite('orig.png', img)


def hough_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
    cv2.imwrite('edge.png', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 开始划线
    cv2.imwrite("image_line.png", image)


def find_rectangle(img):
    # 转换成灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 10, 7, 21)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    cv2.imwrite('thresh.png', thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_areas = [[cnt, cv2.contourArea(cnt)] for cnt in contours if cv2.contourArea(cnt) > 10]
    areas = [cnt_area[1] for cnt_area in cnt_areas]
    contours = [cnt_area[0] for cnt_area in cnt_areas]
    mean_area = np.mean(areas)
    area_mask = np.logical_and(mean_area * 0.5 < np.array(areas), mean_area * 1.2 > np.array(areas))
    contours = np.array(contours)[np.where(area_mask)]
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        warped_img = warp_rect_img(rect, img_gray)
        cv2.imwrite("warp_%d.png" % i, warped_img)
        arr_mean = np.mean(warped_img)
        # 求方差
        arr_var = np.var(warped_img)
        hist = cv2.calcHist([warped_img], [0], None, [256], [0, 256])
        print(hist)
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imwrite("countors.png", img)

    print('ss')


def json_2_mask_img(json_file, save_dir):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    h, w = json_dict['imageHeight'], json_dict['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)
    polygons = [np.array([np.array(point, dtype=np.int32) for point in item['points']]) for item in json_dict['shapes']]

    cv2.fillPoly(mask, polygons, 1)

    save_file = os.path.join(save_dir, os.path.split(json_file)[-1].replace('json', 'jpg'))
    cv2.imwrite(save_file, mask)



if __name__ == '__main__':
    img_path = '/media/chen/wt/data/ocr_data/single_img_test/1/5d9f27d85c50d2198641cc65_1.png'
    # save_path = '/media/chen/wt/data/ocr_data/single_img_test/1/5d9f27d85c50d2198641cc65_1_grey.png'
    # color2gray(img_path, save_path)
    # img_path = r'D:\\data\\test\\'
    # for img_file in os.listdir(img_path):
    img_file = r'DJI_0177_111R.png'
    img = cv2.imread(img_file)
    # ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('cnt', thresh)
    # cv2.waitKey(0)
    find_rect_corner(img)
    # hough_line(img)
    rect = find_rectangle(img)
    root_dir = 'D:\\data\\test\\polar_data\\'
    imgs = glob.glob(root_dir + r"*.JPG")
    jsons = glob.glob(root_dir + r"*.json")
    for img_file, json_file in zip(imgs, jsons):
        json_2_mask_img(json_file, './')
