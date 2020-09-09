import re
import requests
import base64
import numpy as np
import cv2
from urllib import parse
import os
import tqdm
from collections import Counter
import json
import glob
from nltk.tokenize import sent_tokenize
import threading
import time
import sys
import requests
import ctypes
import inspect
from functools import reduce


def get_url_from_file(file):
    urls = []
    with open(file) as f:
        for line in f.readlines():
            urls.extend(re.findall(r'(https?://\S+)', line))
    return list(set(urls))


def get_image_by_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    img_bytes = response.content
    im_base64 = base64.b64encode(img_bytes)
    im_str = str(im_base64, encoding="utf-8")
    im_byte = base64.b64decode(im_str)
    im_arr = np.fromstring(im_byte, np.uint8)
    return cv2.imdecode(im_arr, cv2.IMREAD_COLOR)


def save_image_from_urls(urls, save_path):
    img_counter = Counter()
    for url in urls:
        parse_result = parse.urlparse(url.split('@', 1)[0])
        img_name = os.path.split(parse_result.path)[-1]
        img_counter[img_name] += 1
        img, ext = os.path.splitext(img_name)
        img_index_name = img + '_' + str(img_counter[img_name]) + ext
        color_img = get_image_by_url(url)
        if color_img is not None:
            cv2.imwrite(os.path.join(save_path, img_index_name), color_img)
    return img_counter


def get_images_from_log_file(file, save_path):
    urls = get_url_from_file(file)
    save_image_from_urls(urls, save_path)


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False


def get_images_from_json(file, save_path):
    with open(file) as f:
        lines = f.readlines()
        urls_list = [json.loads(line)['ocr_url'] for line in lines]
        shot_imgs = [json.loads(line)['screen_shot_imgs'] for line in lines]
        err_types = [json.loads(line)['problem_code'] for line in lines]
    # urls = reduce(lambda x, y: x + y, urls_list)
    assert len(urls_list) == len(shot_imgs)
    print("gogo")
    for urls, shot_img, err_type in tqdm.tqdm(zip(urls_list, shot_imgs, err_types)):
        sub_save_path = os.path.join(save_path, str(err_type))
        make_dir(sub_save_path)
        img_counter = save_image_from_urls(urls, sub_save_path)
        # assert len(img_counter) == 1
        # assert len(shot_img) == 1
        for img_name in img_counter:
            img, ext = os.path.splitext(img_name)
            for i, shot in enumerate(shot_img):
                shot_color_img = get_image_by_url(shot)
                if shot_color_img is not None:
                    cv2.imwrite(os.path.join(sub_save_path, img + "_shot_%d" % (i + 1) + ext), shot_color_img)


def deal_vocab_for_print_ocr(vocab_file, labels_file, freq_file):
    vocab = set([c for c in open(vocab_file).readline()])
    with open(labels_file) as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm.tqdm(lines):
            _, label = line.split(' ', 1)
            c_set = set([c for c in label[:-1]])
            if c_set - vocab:
                continue
            new_lines.append(line)
    with open('/media/wt/print_ocr_data/label_1220_new_val.txt', 'w') as f:
        f.writelines(new_lines)
    print(vocab)


def extract_chinese(label_file, pattern):
    with open(label_file) as f:
        line = f.readline()
    zh = [item.strip() + '\n' for item in pattern.split(line) if len(item.strip()) > 2]
    return zh


def extract_english(label_file, pattern_del, pattern_sub):
    with open(label_file) as f:
        line = f.readline()
    en = []
    en_json = json.loads(line)
    for q_id in en_json:
        sent_tokenize_list = sent_tokenize(en_json[q_id]['question'])
        en.extend([re.sub(pattern_del, "", line) for line in sent_tokenize_list])
        for opt in en_json[q_id]['opts']:
            en.extend([re.sub(pattern_del, "", line) for line in opt])
    en = [re.sub(pattern_sub, ' ', line) for line in en if len(line.split(" ")) > 2]
    return [re.sub("  +", ' ', line) + '\n' for line in en]


def generate_lm_corpus(file_dir):
    json_files = glob.glob(file_dir + "/*.json")
    zh_lines = []
    en_lines = []
    p_cn = re.compile(r'[^\u4e00-\u9fa5]')
    p_en_del = re.compile(r"[\u4e00-\u9fa5]")
    p_en_sub = re.compile(r"[，。；：！？“”（）]")
    for file in json_files:
        if "英语" not in file:
            zh_lines.extend(extract_chinese(file, p_cn))
        else:
            en_lines.extend(extract_english(file, p_en_del, p_en_sub))
    with open(os.path.join(file_dir, "zh_lm_corpus.txt"), 'w') as f:
        f.writelines(zh_lines)
    with open(os.path.join(file_dir, "en_lm_corpus.txt"), 'w') as f:
        f.writelines(en_lines)
    return zh_lines, en_lines


class GetImageThread(threading.Thread):
    def __init__(self, url):
        threading.Thread.__init__(self)
        self.url = url
        self.result = None
        self.stopped = False
        # self.timeout = 1

    def run(self):
        self.result = requests.get(self.url)
        self.stopped = True

    def get_result(self):
        return self.result

    def is_stopped(self):
        return self.stopped


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def get_image_with_timeout(image_url, timeout=10):
    image_get_th = GetImageThread(image_url)
    image_get_th.start()
    image_get_th.join(timeout)
    if not image_get_th.is_stopped():
        print("download image from url timeout")
        try:
            _async_raise(image_get_th.ident, SystemExit)
        except Exception:
            print("Warning: thread exit failed")
        else:
            print("all thread done")
    else:
        print("download image from url success")
    return image_get_th.get_result()

if __name__ == '__main__':
    # get_images_from_json('/media/chen/wt/data/ocr_test/ocr_pro_20191209',
    #                      '/media/chen/wt/data/ocr_test/feedback_images')
    # deal_vocab_for_print_ocr('/media/wt/print_ocr_data/label_shrink_train_data_1220_vocab.txt',
    #                          '/media/wt/print_ocr_data/label_1009_new_val.txt',
    #                          '/media/wt/print_ocr_data/label_shrink_train_data_1220_freq.txt')
    # generate_lm_corpus("/media/chen/wt/tools/lm_corpus")
    url = "https://codeload.github.com/shibing624/pycorrector/zip/master"
    # url = "https://ks3-cn-beijing.ksyun.com/yx-yuejuan/2749931/0/189/000171/EE85D93A-321A-11EA-B908-448A5B4F7197.png@base@tag=imgScale&w=1404&h=244&cox=1783&coy=189&c=1&f=1?AccessKeyId=AKLTu3UeHldJSaGxjdNRU3MomQ&Expires=1579397939&Signature=jqvfE+5W3IIcE4HV6NoksL5gCQQ="
    cou = 0
    while True:
        try:
            rel = get_image_with_timeout(url, 2)
            if not rel:
                raise requests.exceptions.Timeout
            else:
                break
        except requests.exceptions.Timeout:
            print("request timeout")
        cou += 1
        print("epoch %d" % cou, type(rel))
    print("end")

