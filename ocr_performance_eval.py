import requests
import json
from ocr_assess_2gz_wt import deal_score
import numpy as np
import re
from collections import Counter
import glob
import os
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import datetime
import base64
from img_process import decode_image_with_base64, visualize, img2base64
import cv2
import PIL.ImageFont as ImageFont
from PIL import Image

url = "http://kb.yunxiao.com/ocr_api/v2/ocr_problem/list?api_key=iyunxiao_tester"
colors = ['blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
          'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
          'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
          'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink',
          'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
          'ghostwhite', 'gold', 'goldenrod', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred',
          'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
          'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon',
          'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
          'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
          'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
          'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
          'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred',
          'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue',
          'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
          'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet',
          'wheat', 'yellow', 'yellowgreen']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
subjects = {'Bio': '生物', 'Chem': '化学', 'Geo': '地理', 'His': '历史', 'Math': '数学',
            'Phy': '物理', 'Pol': '政治', 'Chi': '语文', 'Eng': '英语', 'Unk': 'Unknown'}
problem_code_str = ','.join([str(i) for i in range(1, 22)])


def get_post_result(image_url, request_url, subject=None, img_type='url'):
    if img_type == 'url':
        dict_str = json.dumps({"im_url": image_url, "subject": subject})  # post图片的url地址
    else:
        dict_str = json.dumps(
            {"im_base64": img2base64(image_url), "subject": subject, "type": "ctb"})  # post图片的base64编码

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


def get_online_rec_rel(url, limit, offset, start_time, end_time, plat='mkp'):
    request_body = {
        'limit': limit,
        'offset': offset,
        'plat_from': plat,
        'problem_code': problem_code_str,
        'status': 'all',
        'start_time': start_time,
        'end_time': end_time,
        'get_and_read': 'false',
    }
    response = requests.get(url, params=request_body, timeout=5)
    if response.status_code != 200:
        raise IOError
    ret_dict = json.loads(response.text)
    if ret_dict['code'] != 0:
        raise OSError
    return ret_dict['data']['total'], ret_dict['data']['list']


def text_pre_process(text, subject=None):
    while re.search(r"<(\w).*?>(.*?)</\1.*?>", text):
        text = re.sub(r"<(\w).*?>(.*?)</\1.*?>", r"\2", text)
    text = re.sub(r"<img (src|class)=.+?>", '', text)
    text = re.sub(r"&nbsp;", " ", text)

    # 去掉换行标志
    # text = re.sub(r"\$.+?\$", '', text)

    text = re.sub(r"(<br>)+", " " if subject == '英语' else '', text)

    # 去掉题号和选项号
    text = re.sub(r"(^[0-9A-H]+) *[．\.] *", r"", text)
    # text = re.sub(r"\([0-9]+\)", r"", text)
    text = re.sub(r"[\(（]\d+分[\)）]|([(|（]本题+\d+分+[）|)])", r"", text)
    text = re.sub(r"【题文】", r"", text)
    text = re.sub(r"^[一|二|三|四|五|六|七|八|九]+[|.|．｜、]", r"", text)
    text = re.sub(r"\\rm", r"", text)

    # 多余的下划线统一成一个
    text = re.sub(r"_+", r"_", text)

    return text


def cal_iou_height(box1, box2):
    if box1[3] <= box1[1] or box2[3] <= box2[1]:
        return -1
    y_top_max = max(box1[1], box2[1])
    y_bot_min = min(box1[3], box2[3])
    inter_h = max(y_bot_min - y_top_max, 0)
    union_h = box1[3] - box1[1] + box2[3] - box2[1] - inter_h
    iou = inter_h / union_h
    return iou


def underline_bracket_merge(boxes_coord_t, labellist_t, subject):
    """
    按照文本之间的下划线和括号合并文本行
    :param boxes_coord_t:
    :param labellist_t:
    :param subject:
    :return:
    """
    if len(boxes_coord_t) == 0 or len(labellist_t) == 0 or (
            len(boxes_coord_t) != len(labellist_t)):
        return boxes_coord_t, labellist_t

    underline = "_" * 15 if "语文" == subject else "_" * 8
    spaces = '' if "英语" == subject else ' ' * 8

    boxes_coord_regroup = [boxes_coord_t[0]]
    label_list_regroup = [re.sub('^_+|_+$|__+', underline, labellist_t[0])]
    for label, box in zip(labellist_t[1:], boxes_coord_t[1:]):
        box_pre = boxes_coord_regroup[-1]
        label_pre = label_list_regroup[-1]
        # print(label_pre)
        if cal_iou_height(box_pre, box) > 0.5 and (
                (label_pre != '' and label_pre[-1] == '_') or (label != '' and label[0] == '_')):
            # 两个框在同一高度，并且断点两头有一个是下划线，就可以直接拼起来
            new_label = re.sub('_+$', '', label_pre) + underline + re.sub('^_+', '', label)
            new_label = re.sub('^_+|_+$|__+', underline, new_label)  # 将句首句尾的下划线、句中的连续下划线格式化。
            if subject == '英语':
                # 下划线后如果是标点，不空格，如果不是标点，空一格
                new_label = re.sub(r"(_+) +(\W)", r"\1\2", new_label)
                new_label = re.sub(r"(_+)([^\W_])", r"\1 \2", new_label)
            label_list_regroup[-1] = new_label
            boxes_coord_regroup[-1] = box_pre[:2] + box[2:]
        elif cal_iou_height(box_pre, box) > 0.5 and label_pre != '' and label != '' and (
                (label_pre[-1] == '（' and label[0] == '）') or
                (label_pre[-1] == '[' and label[0] == ']') or
                (label_pre[-1] == '【' and label[0] == '】')):
            if subject == "生物" and label_pre[-1] == '[':
                spaces = ' ' * 5
            new_label = label_pre + spaces + label

            # 将句首句尾的下划线、句中的连续下划线格式化。
            new_label = re.sub('^_+|_+$|__+', underline, new_label)

            label_list_regroup[-1] = new_label
            boxes_coord_regroup[-1] = box_pre[:2] + box[2:]
        else:
            # 将句首句尾的下划线、句中的连续下划线格式化。
            new_label = re.sub('^_+|_+$|__+', underline, label)
            boxes_coord_regroup.append(box)
            label_list_regroup.append(new_label)
    return boxes_coord_regroup, label_list_regroup


def ocr_online_eval(start_time, end_time, offline_url=None, formula_url=None, save_dir=None):
    plats = ['mkp', 'drm', 'kbp', 'mt']

    def download_pic(pic_url):
        header = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "User-Agent": "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                          " Chrome/83.0.4103.61 Safari/537.36",
            "DNT": "1",
            "Pragma": "no-cache",
        }
        response = requests.get(pic_url, timeout=5, headers=header)
        im_base64 = base64.b64encode(response.content)
        im_base64 = str(im_base64, encoding="utf-8")
        code, img = decode_image_with_base64(im_base64)
        if code != 0:
            raise requests.exceptions.ConnectionError
        return img

    lev_counter = Counter()
    offline_lev_counter = Counter()
    limit = 100
    total = 0
    rel_list = []
    for plat in plats:
        plat_total, plat_rel_list = get_online_rec_rel(url, limit, 0, start_time, end_time, plat)
        # t1, _ = get_online_rec_rel(url, limit, 0, start_time, end_time, plat)
        total += plat_total
        offset = limit
        while plat_total > offset:
            _, rel = get_online_rec_rel(url, limit, offset, start_time, end_time, plat)
            plat_rel_list.extend(rel)
            offset += limit
        rel_list.extend(plat_rel_list)

    assert len(rel_list) == total

    recg_statistics = {}
    recg_text_list = []
    date_valid_count = 0
    saved_count = 0
    vis_count = 0
    rec_time_invalid_count = 0
    img_name_set = set()
    for rel in tqdm.tqdm(rel_list):
        rec_date = rel['rec_time'].split('T')[0]
        if rec_date < start_time or rec_date > end_time:
            rec_time_invalid_count += 1
            continue

        rec_month = int(rec_date.split('-')[1])
        if rec_month not in recg_statistics:
            recg_statistics[rec_month] = {}

        if 'attachment' in rel.keys() and rel['attachment'] and 'question' in rel['attachment']:
            question = rel['attachment']['question']
            # orig_text = question['description'] + ' '
            #
            # for stem in question['blocks']['stems']:
            #     options = []
            #     if 'options' in stem:
            #         options = stem['options']
            #     orig_text += stem['stem'] + " " + " ".join([k + '．' + options[k] for k in options])

            # text = text_pre_process(orig_text, question['subject']).strip()
            # recg_text = text_pre_process(rel['rec_text'], question['subject']).strip()
            # score, norm_score, text_rel, text_recg = deal_score(text, recg_text, question['subject'], delsym=delsym)

            # lev_counter[score] += 1

            # recg_statistics[rec_month][question['subject']]['lev'][score] += 1
            # valid_count += 1
        else:
            if 'subject' in rel:
                question = {'subject': rel['subject']}
            else:
                question = {'subject': "Unknown"}
            # print("em....")

        if question['subject'] not in recg_statistics[rec_month]:
            recg_statistics[rec_month][question['subject']] = {'count': 0, 'lev': Counter()}
        recg_statistics[rec_month][question['subject']]['count'] += 1

        if rel['problem_code'] not in recg_statistics[rec_month]:
            recg_statistics[rec_month][rel['problem_code']] = {'count': 0, 'lev': Counter()}
        recg_statistics[rec_month][rel['problem_code']]['count'] += 1
        # recg_statistics[rec_month][rel['problem_code']]['lev'][score] += 1
        img_dir = os.path.join(os.path.join(save_dir, question['subject']),
                               '%s_%d' % (re.sub('（.+）', '', rel['problem_type']), rel['problem_code']))
        date_valid_count += 1
        if save_dir:
            pic_url = rel['ocr_url'][0]
            try:
                pic_name = re.search(r'[\w-]+?(\.jpg|\.png|\.jpeg)', pic_url).group()
                img_index_suffix = 0
                unique_pic_name = pic_name
                while unique_pic_name in img_name_set:
                    unique_pic_name = re.sub(r'(\.jpg|\.png|\.jpeg)', r'-%d\1' % img_index_suffix, pic_name)
                    img_index_suffix += 1
                pic_name = unique_pic_name
                # if pic_name != '36ABDC30-A3EB-11EA-9BCC-F875A42A7A98.png':
                #     continue
                img_name_set.add(pic_name)

            except:
                print("\nWarning: pic_url invalid\n", pic_url)
                continue
            # else:
            #     continue
            if len(rel['ocr_url']) > 1:
                # print('\n========')
                # [print(u) for u in rel['ocr_url']]
                retry_count = 3
                while retry_count > 0:
                    retry_count -= 1
                    try:
                        imgs = [download_pic(u) for u in rel['ocr_url']]
                        break
                    except Exception as e:
                        print(e, rel['ocr_url'])
                        # continue
                else:
                    continue
                shapes = np.array([img.shape for img in imgs])
                max_w = np.max(shapes, 0)[1]
                total_h = np.sum(shapes, 0)[0]
                img = np.ones([total_h, max_w, imgs[0].shape[2]]) * 255
                h = 0
                for item in imgs:
                    img[h:item.shape[0] + h, :item.shape[1], :] = item
                    h += item.shape[0]
                # continue
            else:
                try:
                    img = download_pic(pic_url)
                except Exception as e:
                    print(e, pic_url)
                    continue
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            img_path = os.path.join(img_dir, pic_name)
            try:

                pil_img = Image.fromarray(img.astype(np.uint8))
            except TypeError as e:
                print(e)
                continue

            pil_img.save(img_path)
            # img_path = './img\\2020-07-29\\' + pic_name
            # cv2.imwrite(img_path, img)


            recg_text_list.append(os.path.join(img_dir, pic_name) + ' ' + rel['rec_text'] + '\n')

            # if len(rel['screen_shot_imgs']) != 0:
            #     shot_url = rel['screen_shot_imgs'][0]
            #     _, shot_img = download_pic(shot_url)
            #     cv2.imwrite(os.path.join(img_dir, re.sub(r'\.(jpg|png|\.jpeg)', r'_1.\1', pic_name)), shot_img)
            for i, shot_url in enumerate(rel['screen_shot_imgs']):
                try:
                    shot_img = download_pic(shot_url)
                except:
                    continue
                cv2.imwrite(os.path.join(img_dir, re.sub(r'\.(jpg|png|jpeg)', r'_%d.\1' % (i + 1), pic_name)),
                            shot_img)

            if offline_url:
                try:
                    data = get_post_result(img_path, offline_url, question['subject'], 'base64')
                except requests.exceptions.ConnectionError:
                    data = None
                # _, labellist = underline_bracket_merge(data['boxes_coord'], data['labellist'],
                #                                        question['subject'])
                if not data:
                    continue
                boxes_coord = data.get("boxes_coord")
                if not boxes_coord:
                    print("Nothing can be detected!")
                    continue
                labellist = data.get("labellist")
                visualize(os.path.join(img_dir, pic_name),
                          re.sub(r'\.(jpg|png|jpeg)', r'_0.\1', os.path.join(img_dir, pic_name)), boxes_coord,
                          labellist, font,
                          subject=question['subject'],
                          vis_image=True)  # 5.检测与识别结果可视化
                if formula_url and question['subject'] in ['数学', '物理', '化学', '生物']:
                    try:
                        data = get_post_result(img_path, formula_url, question['subject'], 'base64')
                    except requests.exceptions.ConnectionError:
                        data = None
                    if not data:
                        continue
                    boxes_coord = data.get("boxes_coord")
                    if not boxes_coord:
                        print("Nothing can be detected!")
                        continue
                    labellist = data.get("labellist")
                    visualize(os.path.join(img_dir, pic_name),
                              re.sub(r'\.(jpg|png|jpeg)', r'_formula.\1', os.path.join(img_dir, pic_name)), boxes_coord,
                              labellist, font,
                              subject=question['subject'],
                              vis_image=True)  # 5.检测与识别结果可视化
                vis_count += 1
            saved_count += 1
    print(len(img_name_set) == saved_count)
    print("img_name_set: %d" % len(img_name_set), "saved_count: %d" % saved_count,
          "vis_count: %d" % vis_count,
          "rec_time_invalid_count: %d" % rec_time_invalid_count)
    print("=================================================")
    print("total feedback count: %d, valid: %d, saved: %d" % (len(rel_list), date_valid_count, saved_count))
    print("=================================================")
    if save_dir and os.path.exists(save_dir):
        with open(os.path.join(save_dir, 'online_recg.txt'), 'w') as f:
            f.writelines(recg_text_list)
    return recg_statistics


subject_name_en2cn = {'biology': '生物', 'chemistry': '化学', 'geography': '地理', 'history': '历史', 'math': '数学',
                      'physics': '物理', 'politics': '政治', 'chinese': '语文', 'english': '英语', 'barcode': 'barcode'}


def eval_by_txts(subject_dir):
    lev_counter = Counter()
    scores = []
    total = 0
    discard = 0
    for dir in os.listdir(subject_dir):
        subject = os.path.split(dir)[1].split('_')[1]
        txt_files = glob.glob(os.path.join(subject_dir, dir) + "/*.txt")
        for file in txt_files:
            with open(file) as f:
                lines = f.readlines()
                gt = lines[0]
                rec = lines[1]
                score, text_rel, text_recg = deal_score(gt, rec, subject, delsym=True)
                if score > 20:
                    discard += 1
                    continue
                lev_counter[score] += 1
                total += 1
                scores.append(score)
    # lev_counter = sorted(lev_counter)
    print(np.average(scores) if scores != [] else "\n===========\nNothing got\n")
    print(sorted(lev_counter.items(), key=lambda item: item[0]))
    print("All Correct:", lev_counter[0], '/', total)
    print("Discard:", discard)


def compare_with_offline(offline_url):
    start = '2020-2-1'
    end = '2020-2-10'
    recg_statistics = ocr_online_eval(start, end, offline_url)
    print(recg_statistics)


def single_img_offline_recg(url, offline_url):
    text_offline = ""
    data = get_post_result(
        url,
        offline_url, "语文"
    )
    text_offline += "".join(data["labellist"])
    print(text_offline)


def histogram_vis(save_name, num_lists, x_label_list, x_label, y_label, title="2020 OCR online feedback statistics"):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    x = range(len(x_label_list))
    column_width = 1
    max_height = 0
    # num_lists = num_lists * 12
    column_count = len(num_lists)
    column_space = 0.8 * column_width
    for i, num_list in enumerate(num_lists):
        rects = plt.bar([((column_count + column_space) * j + i) * column_width for j in x], height=num_list,
                        width=column_width,
                        color=colors[i], label=months[i])
        max_height = max(max_height, max(num_list))
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + column_width / 2, height, ("%.2f" % height) if y_label != 'num' else str(height),
                     ha="center", va="bottom")

    # plt.figure(figsize=(6, 9))
    plt.ylim(0, int(max_height * 1.3))  # y轴取值范围
    plt.ylabel(y_label)

    plt.xticks([(index * (column_count + column_space) + (column_count - 1) / 2) * column_width for index in x],
               x_label_list)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()  # 设置题注
    plt.savefig(save_name)
    plt.show()
    plt.close()


def vis_online():
    start = '2020-08-01'
    today = datetime.date.today()
    # offline_url = "http://192.168.0.229:10086/get_ocr_result_v2"
    offline_url = "http://yj-ocr.haofenshu.com/get_ocr_result_v2"
    formula_url = "http://yj-mathpix.haofenshu.com/get_formula_rec"
    # offline_url = ''
    save_dir = './img/' + str(today)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rec_statistics = ocr_online_eval(start, str(today), offline_url=offline_url, formula_url=formula_url,
                                     save_dir=save_dir)
    if len(rec_statistics) == 0:
        return

    problem_code_list = range(1, 22)  # 横坐标刻度显示值

    problem_nums = []  # 按照问题类型统计的反馈个数
    # problem_levs = []  # 按照问题类型统计的平均编辑距离
    subject_nums = []  # 按照科目统计的反馈个数
    # subject_levs = []  # 按照科目统计的平均编辑距离
    for month in rec_statistics:
        # statistics = sorted(month.items(), key=lambda item: item[0])
        problem_num = []
        # problem_lev = []
        for err_code in problem_code_list:
            if err_code not in rec_statistics[month]:
                problem_num.append(0)
                # problem_lev.append(0)
            else:
                problem_num.append(rec_statistics[month][err_code]['count'])
                # lev_count = rec_statistics[month][err_code]['lev']
                # problem_lev.append(
                #     sum([lev * lev_count[lev] for lev in lev_count]) / sum([lev_count[lev] for lev in lev_count]))

        problem_nums.append(problem_num)
        # problem_levs.append(problem_lev)

        subject_num = []
        # subject_lev = []
        for subject in subjects:
            if subjects[subject] not in rec_statistics[month]:
                subject_num.append(0)
                # subject_lev.append(0)
            else:
                subject_num.append(rec_statistics[month][subjects[subject]]['count'])
                # lev_count = rec_statistics[month][subjects[subject]]['lev']
                # subject_lev.append(
                #     sum([lev * lev_count[lev] for lev in lev_count]) / sum([lev_count[lev] for lev in lev_count]))
        subject_nums.append(subject_num)
        # subject_levs.append(subject_lev)

    histogram_vis(os.path.join(save_dir, "problem_code.jpg"), problem_nums, problem_code_list, 'problem_code', 'num')
    # histogram_vis(problem_levs, problem_code_list, 'problem_code', 'lev')
    histogram_vis(os.path.join(save_dir, "subject.jpg"), subject_nums, [k for k in subjects], 'subject', 'num')
    # histogram_vis(subject_levs, [k for k in subjects], 'subject', 'lev')


if __name__ == "__main__":
    font_ttf = "./SourceHanSans-Normal.ttf"  # 可视化字体类型
    font = ImageFont.truetype(font_ttf, 20)  # 字体与字体大小
    # offline_url = "http://192.168.1.229:9111/get_ocr_result_v2"
    # compare_with_offline(offline_url)
    vis_online()
    # eval_by_txts("/media/chen/hzc/datasets/ocr_res")
