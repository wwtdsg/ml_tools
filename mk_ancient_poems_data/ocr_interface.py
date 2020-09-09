# codin: utf-8
import requests
import glob
import cv2
import numpy as np
import operator
import os
import shutil
import re

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from skimage import io
import json
import base64
from urllib import parse

font_ttf = "SourceHanSans-Normal.ttf"  # 可视化字体类型
font = ImageFont.truetype(font_ttf, 20)  # 字体与字体大小
valid_ext = [r".jpg", r".png", r".jpeg"]


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


def get_basename(path):
    """
    根据url决定保存的图片名称
    """
    basename_t = path.split("@")[0]
    basename = os.path.basename(basename_t)
    basename_prefix, ext = os.path.splitext(basename)
    xywh = path.split("&")[-4:]
    # x, y, w, h = xywh
    xywh = list(map(lambda x: int(x.split("=")[1]), xywh))
    x, y, w, h = xywh
    new_basename = "{}_x{}_y{}_w{}_h{}{}".format(basename_prefix, x, y, w, h, ext)

    return new_basename


def cv2ImgAddText(img, font, text, left, top, textColor=(255, 0, 0), textSize=20):
    text = " " if text == "" else text
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((left, top), text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def visualize_one_line(image_path, pred, gt, vis_path, font):
    _, name = os.path.split(image_path)
    save_image_file = "{}{}.jpg".format(vis_path, name)
    img = cv2.imread(image_path)
    img_dst = np.ones((img.shape[0] + 120, img.shape[1] + 100, img.shape[2]), dtype=np.uint8) * 255
    img_dst[:img.shape[0], :img.shape[1], :] = img
    img_dst = cv2ImgAddText(img_dst, font, "{}".format(pred[0]), 0, img.shape[0], textSize=20)
    if len(pred) > 1:
        img_dst = cv2ImgAddText(img_dst, font, "{}".format(pred[1]), 0, img.shape[0] + 20, textSize=20)
    if len(pred) > 2:
        img_dst = cv2ImgAddText(img_dst, font, "{}".format(pred[2]), 0, img.shape[0] + 40, textSize=20)
    img_dst = cv2ImgAddText(img_dst, font, "{}".format(gt), 0, img.shape[0] + 60, textSize=20, textColor=(255, 0, 255))
    cv2.imwrite(save_image_file, img_dst)


def visualize(im_name, boxes_coord, labellist, out_dir, font, subject=None, path_perfect=False, vis_image=True):
    """
    检测与识别结果可视化.
    :param im_name: 图片路径/图片url
    :param boxes_coord: 检测框位置信息[[xmin, ymin, xmax, ymax],...]
    :param labellist: 检测框识别结果字符串[res_string1, res_string2]
    :param out_dir: 结果写出目录
    :param font: 字体类型及大小
    :param subject: 科目类别
    """
    # print('#############')
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

            print('vising', out_dir)

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
        path = parse.urlparse(im_name)[2]
        basename = get_basename(path)  # 06F43B89-BB73-11E6-9864-D02788899015_x357_y2735_w2259_h399.png
    if not path_perfect:
        out_dir = os.path.join(out_dir, str(subject))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base_name = basename
    # base_name = "_".join(im_name.split("/")[-3:])
    stem, ext = os.path.splitext(base_name)
    txt_basename = stem + "_gz_text.txt"
    # if  path_perfect:
    #     txt_basename= os.path.basename(txt_basename)
    #     base_name = os.path.basename(base_name)
    txt_basename = os.path.basename(txt_basename)
    base_name = os.path.basename(base_name)

    print(txt_basename, base_name)
    fw = open(os.path.join(out_dir, txt_basename), "w", encoding='utf-8')
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
            # fw.write("{}, {}, {}, {}, {}\n".format(xmin, ymin, xmax, ymax, labellist[idx].encode("utf-8")))    # python2
            fw.write("{}, {}, {}, {}, {}\n".format(xmin, ymin, xmax, ymax, labellist[idx]))  # python3
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
                # fw.write("{}, {}, {}, {}, {}\n".format(xmin, ymin, xmax, ymax, linelabels[idx].encode("utf-8")))   # python2
                fw.write("{}, {}, {}, {}, {}\n".format(xmin, ymin, xmax, ymax, linelabels[idx]))  # python3

            # txt_draw.text((im_w + 10, count * 42 + 20), linelabel, fill=6, font=font)
            if vis_image:
                txt_draw.text((draw_start_x, curr_y), linelabel, fill=6, font=font)
                label_w, label_h = font.getsize(linelabel)  # 返回的是w,h
                curr_x += label_w
                curr_y += label_h
    fw.close()
    if vis_image:
        newimg.save(os.path.join(out_dir, base_name))


def get_post_result(image_url, request_url, style="im_url", subject=None):
    try:
        # print(image_url)
        img = io.imread(image_url)
    except Exception as e:
        print(e)
        return
    if img is None:
        return

    # h, w = img.shape[:2]
    if style == "im_url":
        # start = time.time()
        dict_str = json.dumps({"im_url": image_url, "subject": subject})  # post图片的url地址
    elif style == "base64":
        dict_str = json.dumps({"im_base64": img2base64(image_url), "subject": subject})  # post图片的base64编码
    elif style == "base64_list":
        dict_str = json.dumps({"im_base64_list": img2base64(image_url), "subject": subject})  # post图片的base64编码

    # ret_dict = get_ocr_result(dict_str)  # 2.将base64编码传入接口
    try:
        response = requests.post(request_url, data=dict_str)
    except Exception as e:
        print(e)
        raise Exception
    if response.status_code != 200:
        raise Exception
        return

    ret_dict = json.loads(response.text)
    if not ret_dict:
        print("get response data missing or image too large!")
        return

    status = ret_dict.get("code")  # 3.判断响应状态是否正常
    if status:
        print(ret_dict)
        return

    # 4.返回检测结果，识别结果
    data = ret_dict.get("data")
    data = json.loads(data) if data is not None else None
    return data


def test_ocr(image_url, request_url, style="im_url", subject=None, save_dir=None, path_perfect=False):
    # request_url = "http://yj-ocr.haofenshu.com/get_ocr_result_v2"
    # subject = '数学'
    data = get_post_result(image_url, request_url, style, subject)
    boxes_coord = data.get("boxes_coord")
    if not boxes_coord:
        print("Nothing can be detected!")
        return
    labellist = data.get("labellist")
    # print('before vis',out_dir)
    visualize(image_url, boxes_coord, labellist, save_dir, font, subject=subject,
              path_perfect=path_perfect, vis_image=True)  # 5.检测与识别结果可视化
    # print('after vis')
    return


def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False


def test_only_subject(ds_dir, request_url, src_dir_name, save_dir_path, subject_name_en2cn):
    #############
    ## with rank dataset test
    #############
    subject_dirs = ['senior_math', 'senior_chemistry', 'senior_physics', 'senior_biology']
    # for subject_dir_name in os.listdir(os.path.join(ds_dir, src_dir_name)):
    for subject_dir_name in subject_dirs:
        # subject_dir_name = 'junior_geography'
        # subject_dir_name = 'junior_chinese'
        save_subject_dir_path = os.path.join(save_dir_path, subject_dir_name)
        if not makeDir(save_subject_dir_path):
            pass

        cur_dst_dir = os.path.join(ds_dir, src_dir_name, subject_dir_name)

        # pool = Pool(3)
        for img_path in os.listdir(cur_dst_dir):
            _, ext = os.path.splitext(img_path)
            if ext not in valid_ext:
                continue
            img_path = os.path.join(cur_dst_dir, img_path)
            # img_path = "//media/chen/wt/tools/img/2020-04-27/语文/符号错漏_7/5e958b0dfa0eb6080527d4f7.png"
            #         "formula_detect_err_data/problem_data/ori/senior_math/139594_20.png"
            # for img_path in glob.glob(os.path.join(cur_dst_dir, '*.png')):
            # print(img_path)
            img_name = os.path.basename(img_path)
            img_name_no_suffix = img_name.split('.')[0]
            # print(img_name_no_suffix)

            txt_paths = glob.glob(os.path.join(cur_dst_dir, img_name_no_suffix + '_text.txt'))
            for txt in txt_paths:
                shutil.copy(txt, save_subject_dir_path)

            bd_txt_paths = glob.glob(os.path.join(cur_dst_dir, img_name_no_suffix + '_bd_text.txt'))
            for txt in bd_txt_paths:
                shutil.copy(txt, save_subject_dir_path)

            # test_ocr(img_path, "base64", subject_name_en2cn.get(subject_dir_name.split('_')[-1], ''),
            test_ocr(img_path, request_url, "base64", subject_name_en2cn.get(subject_dir_name.split('_')[-1], ''),
                     save_subject_dir_path, True)
        # break


# add by lpj
def test_subject_with_rank(ds_dir, src_dir_name, save_dir_path, subject_name_en2cn):
    #############
    ## with rank dataset test
    #############
    for subject_dir_name in os.listdir(os.path.join(ds_dir, src_dir_name)):
        save_subject_dir_path = os.path.join(save_dir_path, subject_dir_name)
        makeDir(save_subject_dir_path)
        for subject_dir_path, subject_subdir_names, _ in os.walk(os.path.join(ds_dir, src_dir_name, subject_dir_name)):
            # print(subject_dir_path, subject_subdir_names, _ )
            for subject_subdir_name in subject_subdir_names:  # 0.0~0.9
                # makeDir(os.path.join())
                cur_dst_dir = os.path.join(subject_dir_path, subject_subdir_name)
                save_subject_subdir_path = os.path.join(save_subject_dir_path, subject_subdir_name)
                makeDir(save_subject_subdir_path)

                # pool = Pool(3)
                for img_path in glob.glob(os.path.join(cur_dst_dir, '*.png')):
                    # print(img_path)
                    img_name = os.path.basename(img_path)
                    img_name_no_suffix = img_name.split('.')[0]
                    # print(img_name_no_suffix)

                    txt_paths = glob.glob(os.path.join(cur_dst_dir, img_name_no_suffix + '_text.txt'))
                    for txt in txt_paths:
                        shutil.copy(txt, save_subject_subdir_path)

                    bd_txt_paths = glob.glob(os.path.join(cur_dst_dir, img_name_no_suffix + '_bd_text.txt'))
                    for txt in bd_txt_paths:
                        shutil.copy(txt, save_subject_subdir_path)

                    test_ocr(img_path, "base64", subject_name_en2cn.get(subject_dir_name.split('_')[-1], ''),
                             save_subject_subdir_path, True)


def test_recg_only_by_img(img_path):
    request_url = "http://192.168.1.229:8112/get_crnn_result"

    # image_name, label = line.strip().split(' ', 1)
    _, ext = os.path.splitext(img_path)
    data = get_post_result(img_path, request_url, "base64_list")
    if data is None:
        raise Exception("oops!")
    pred = data.get("pred_str_list")
    print(pred)


def test_recg_only():
    request_url = "http://192.168.0.229:19232/get_crnn_result"

    # label_path = '/media/chen/lpj/dataset/data4poem/labels/cleaned_label_1-30_shuf_val.txt'
    label_path = '/media/chen2/data/wt/hw_poem_true_negative/hw_poem_true_negative_labels.txt'
    data_root = ''

    vis_dir = '/media/chen/wt/data/hw_poem_true_negative_test/'
    # label_path = '/media/chen/lyh/datas/test_data/label.txt'
    # data_root = '/media/chen/lyh/datas/test_data'
    success = 0
    total = 0
    with open(label_path, 'r') as fd:
        for line in fd:
            image_name, label = line.strip().split(' ', 1)

            img_path = os.path.join(data_root, image_name)
            # img_path = "/media/chen/wt/data/hw_poem_true_negative_test/test7.png"
            _, ext = os.path.splitext(img_path)
            if ext not in valid_ext:
                continue
            data = get_post_result(img_path, request_url, "base64_list")
            if data is None:
                raise Exception("oops!")
            pred = data.get("pred_str_list")
            if label.strip() == pred[0]:
                success += 1
            else:
                visualize_one_line(img_path, pred, label.strip(), vis_dir, font)
            total += 1

    print('total: {}  success:{}'.format(total, success))
    print("accuracy:", success * 1.0 / total)


subject_name_en2cn = {'biology': '生物', 'chemistry': '化学', 'geography': '地理', 'history': '历史', 'math': '数学',
                      'physics': '物理', 'politics': '政治', 'chinese': '语文', 'english': '英语', 'barcode': 'barcode'}


def test_print_ocr():
    out_dir = "./res"  # 检测与识别结果可视化目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    request_url = "http://192.168.0.229:19232/get_ocr_result_v2"
    # request_url = "http://yj-ocr.haofenshu.com/get_ocr_result_v2"

    ds_dir = '/media/chen/mcm/ctpn_test_2019_01_23/ocr_KBM_test/'
    # ds_dir = '/media/chen/wt/data/ocr_data/'
    src_dir_name = 'test_data_191115'
    # src_dir_name = 'test_data'
    # src_dir_name = 'feedback_err_data'
    save_dir_name = 'feedback_err_data_chinese'  # 'test_data_vis_rescrnn_degaug_5.27_from0'

    save_dir_path = os.path.join(ds_dir, save_dir_name)
    makeDir(save_dir_path)
    test_only_subject(ds_dir, request_url, src_dir_name, save_dir_path, subject_name_en2cn)
    # test_ocr(ds_dir, src_dir_name, save_dir_path, subject_name_en2cn)


def test_process():
    file = "/media/chen/wt/data/ocr_data/test_data2/junior_english/space_error.png"
    dst_file = "/media/chen/wt/data/ocr_data/test_data2/junior_english/space_error_erosion.png"
    img = cv2.imread(file, 0)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(dst_file, erosion)


def test_feedback_images():
    ds_dir = '/media/chen/wt/data/ocr_data/'
    src_dir_name = 'feedback_err_data'
    save_dir_name = 'feedback_err_data_res_3_12000'  # 'test_data_vis_rescrnn_degaug_5.27_from0'
    save_dir_path = os.path.join(ds_dir, save_dir_name)
    makeDir(save_dir_path)
    # formula_dirs = ['senior_biology', 'junior_biology', 'senior_physics', 'senior_geography']
    for subject_dir_name in os.listdir(os.path.join(ds_dir, src_dir_name)):
        # for subject_dir_name in formula_dirs:
        # subject_dir_name = 'junior_geography'
        # subject_dir_name = 'junior_chinese'
        save_subject_dir_path = os.path.join(save_dir_path, subject_dir_name)
        if not makeDir(save_subject_dir_path):
            pass
        cur_dst_dir = os.path.join(ds_dir, src_dir_name, subject_dir_name)
        for img_path in os.listdir(cur_dst_dir):
            if 'shot' in img_path:
                continue
            prefix, ext = os.path.splitext(img_path)
            if ext not in valid_ext:
                continue
            img_path = os.path.join(cur_dst_dir, img_path)
            test_ocr(img_path, "base64", None,
                     save_subject_dir_path, True)


def test_single_image():
    request_url = "http://yj-ocr.haofenshu.com/get_ocr_result_v2"
    ds_dir = '/media/chen/wt/tools/mk_ancient_poems_data/test_data'
    src_dir_name = 'test_imgs'
    # src_dir_name = 'gen_num_space_1'
    save_dir_name = 'ocr_imgs'  # 'test_data_vis_rescrnn_degaug_5.27_from0'
    save_dir_path = os.path.join(ds_dir, save_dir_name)
    makeDir(save_dir_path)
    for img in os.listdir(os.path.join(ds_dir, src_dir_name)):
        img_path = os.path.join(ds_dir, src_dir_name, img)
        # prefix, ext = os.path.splitext(img_path)
        test_ocr(img_path, request_url, "base64", "语文", save_dir_path, True)


def test_formula_image():
    out_dir = "./res"  # 检测与识别结果可视化目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # request_url = "http://192.168.1.229:11333/get_formula_rec"
    request_url = 'http://yj-mathpix.haofenshu.com/get_formula_rec'
    ds_dir = '/media/chen/mcm/ctpn_test_2019_01_23/ocr_KBM_test/'
    # ds_dir = '/media/chen/wt/data/ocr_data/'
    src_dir_name = 'test_data_191115'
    # src_dir_name = 'test_data'
    # src_dir_name = 'feedback_err_data'
    save_dir_name = 'feedback_err_data_text_formula_coprocess'  # 'test_data_vis_rescrnn_degaug_5.27_from0'

    save_dir_path = os.path.join(ds_dir, save_dir_name)
    makeDir(save_dir_path)
    test_only_subject(ds_dir, request_url, src_dir_name, save_dir_path, subject_name_en2cn)


if __name__ == "__main__":
    #
    # request_url = "http://192.168.1.229:8111/get_crnn_result"

    # test_recg_only()
    # test_recg_only_by_img('/media/chen/wt/data/ocr_data/test_data2/junior_chinese/filter.png')
    # test_process()
    # test_feedback_images()
    # test_print_ocr()
    # test_formula_image()
    test_single_image()
