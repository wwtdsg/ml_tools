# coding=utf-8
"""
python 2.7
"""

# reload(sys)
# sys.setdefaultencoding('utf-8')
import urllib, base64
import json
import re
from PIL import Image, ImageFont, ImageDraw
import os
import time
from urllib import parse
from txt_process import make_dir
import re

time1 = time.time()

def get_token(API_Key, Secret_Key):
    """
    百度ocr每次调用时的 token 获取
    :param API_Key:
    :param Secret_Key:
    :return:
    """
    # access_token
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + API_Key + '&client_secret=' + Secret_Key
    # host = 'https://aip.baidubce.com/rest/2.0/solution/v1/form_ocr/request?grant_type=client_credentials&client_id=' + API_Key + '&client_secret=' + Secret_Key
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    content_json = json.loads(content)
    access_token = content_json['access_token']
    return access_token


def recognition_word_high(filepath, savepath, filename, access_token):
    # url='https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=' + access_token
    # url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token=' + access_token
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=' + access_token
    if 'txt' == filename[-3:]:
        return
    print(filepath)
    f = open(os.path.join(filepath, filename), 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img}
    data = parse.urlencode(params).encode('utf-8')
    request = urllib.request.Request(url, data)
    response = urllib.request.urlopen(request)
    content = response.read().decode()
    # print(response.read().decode())
    # request = urllib.request.Request(url, params)
    # request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    # request = parse.urlencode(request)
    # response = urllib.request.urlopen(request)
    # content = response.read()
    # print content
    if content:
        txtfile = open(os.path.join(savepath, filename.split('.')[0] + '_bd_text.txt'), 'w')
        world = re.findall('"words": "(.*?)"}', str(content), re.S)
        oriimg = Image.open(os.path.join(filepath, filename))
        w, h = oriimg.size
        nw = int(w * 2.5)
        nh = int(60 * len(world) + 20)
        if nh < h:
            nh = h
        # newimg = Image.new("L", (nw, nh), 255)
        # image_font = ImageFont.FreeTypeFont(font='simhei.ttf', size=32)
        # txt_draw = ImageDraw.Draw(newimg)
        num = 0
        line = "".join(world)
        if line == '':
            print("Woop")
            return
        line = re.sub("[0-9a-zA-Z.,]", '', line)
        # for each in world:
        #     # eachstr = each.replace('\\', '').decode('utf-8')
        #     # txt_draw.text((w + 60, 10 + num * 60), eachstr, fill=6, font=image_font)
        #     num += 1
        txtfile.writelines(line + '\n')
        txtfile.close()
        # newimg = newimg
        # newimg.paste(oriimg, (0, 0))
        # newimg.save(os.path.join(savepath, filename))


if __name__ == '__main__':

    API_Key = "9i7h8oofMmMO1aShDewAFnLC"
    Secret_Key = "74dwwOfrppgFVVTi71EyvUaqesL53BhO"
    root_dir = r"/media/chen/mcm/ctpn_test_2019_01_23/ocr_KBM_test/test_data"
    for subject in os.listdir(root_dir):
        filepath = os.path.join(root_dir, subject)
        savepath = r"/media/chen/mcm/ctpn_test_2019_01_23/ocr_KBM_test/test_data/" + subject
        savepath = r"/media/chen/mcm/ctpn_test_2019_01_23/ocr_KBM_test/test_data/" + subject
        files = os.listdir(filepath)
        make_dir(savepath)
        for file in files:
            # print(file)
            # if os.path.exists(os.path.join(savepath, file)):
            #     print(file + ':' + '已识别')
            # else:
            access_token = get_token(API_Key, Secret_Key)
            start = time.time()
            recognition_word_high(filepath, savepath, file, access_token)
            end = time.time()
            print(end - start)
