import os
from glob import glob

# import Levenshtein
import pandas as pd
import argparse
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
# from hanziconv import HanziConv


def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    text = " " if text == "" else text
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "NotoSansCJK-Black.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def vis_img(debug_img_path, text, gz_text, debug_dir, gz_score):
    img = cv2.imread(debug_img_path)

    maxLen = max(len(text), len(gz_text))
    text_size = 20
    nFontline = img.shape[1] // text_size
    nLine = maxLen // nFontline
    nLine += 1 if maxLen % nFontline else 0

    img_dst = np.ones((img.shape[0] + (nLine * 2) * 30, img.shape[1], img.shape[2]),
                      dtype=np.uint8) * 255

    img_dst[:img.shape[0], :img.shape[1], :] = img
    new_text = ''
    new_gz_text = ''

    for i in range(len(text)):
        new_text += text[i]
        if (i + 1) % nFontline == 0:
            new_text += '\n'

    for i in range(len(gz_text)):
        new_gz_text += gz_text[i]
        if (i + 1) % nFontline == 0:
            new_gz_text += '\n'

    img_dst = cv2ImgAddText(img_dst, new_text, 0, img.shape[0], textColor=(255, 0, 0), textSize=text_size)
    img_dst = cv2ImgAddText(img_dst, new_gz_text, 0, img.shape[0] + nLine * 30, textColor=(0, 255, 0),
                            textSize=text_size)
    cv2.imwrite(os.path.join(debug_dir, str(gz_score) + '_' + os.path.basename(debug_img_path)), img_dst)


def get_cur_mon():
    import time
    return time.localtime().tm_mon


def get_aver_score(score_d):
    score_l = 0
    key_l = 'gz_score_l'
    return float(sum(score_d[key_l])) / len(score_d[key_l]) if len(score_d[key_l]) else 0


def print_base_info():
    """用的比较多，封装成函数"""
    print('\t'.join([u'编辑距离：', '0~9', '10~19', '20~29',
                     '30~39', '40~49', '50~59', '60~69',
                     '70~79', '80~89', '90~', ]))


def insert_staic(staic_d):
    """在值的列表的第0个元素插入，为了打印出来"""
    for value in staic_d.values():
        value.insert(0, u'数据数量：')


def create_score_d():
    """创建统计分数字典"""
    score_d = {}
    score_d['gz_score_l'] = []

    return score_d


def create_staic_d():
    """创建统计范围字典"""
    staic_d = {}
    staic_d['gz_score_staic_l'] = [0] * 10
    return staic_d


def get_index(score):
    """获取一条数据编辑距离所在的列表索引"""
    index = (score // 10) if (score // 10) < 9 else 9
    return index


alpha_num_list = '0123456789' \
                 'abcdefghijklmnopqrstuvwxyz' \
                 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ' '


def trans_chi(text, subject):
    """将标点符号去除，连续的非中文替换为1个空格"""
    text = ''.join(text) if isinstance(text, list) else text
    text = re.sub(r"[^\u4e00-\u9fa5]", r"", text) if subject != "英语" else \
        re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5 ]", r" ", text)
    text = re.sub(r" +", r" ", text)
    text = HanziConv.toSimplified(text)
    return text


def _read_content(txt_path, style="else", subject=None):
    """
    读取文本内容
    :param path: 路径
    :return:
    """
    if style == "else":
        with open(txt_path, 'r', encoding="utf-8") as fr:
            content = fr.read()  # <type 'unicode'>

        # if delsym:
        #     content = trans_chi(content)

    elif style == "gz":
        # with codecs.open(txt_path, encoding="gbk") as fr:
        with open(txt_path, 'r', encoding="utf-8") as fr:
            lines = fr.readlines()
        content = []
        for line_no, line in enumerate(lines):
            # if len(line.split(", ", 4)) < 5:
            #     print(txt_path, line)

            line = line.split(", ", 4)[4].strip()
            content.append(line)
        if len(content) > 0:
            content[-1] = content[-1] + "\t"
            # line = line.strip("\n\r")
            # if line_no != len(lines) - 1:
            #     content.append(line + "\t")
            # else:
            #     content.append(line)
        # if delsym:
        #     content = trans_chi(content)
        content = "".join(content) if subject != '英语' else " ".join(content)
    return content


def print_static_info(score_d, staic_d, sub_dir_name):
    """将各种信息按照对应格式打印出来"""
    gz_aver_score = get_aver_score(score_d)
    print('学科:\t%s' % sub_dir_name)
    print('gz_text平均距离:\t%f' % gz_aver_score)
    print_base_info()
    print('\t'.join([str(i) for i in staic_d['gz_score_staic_l']]))
    print('\n样本总量：\t%d' % sum(staic_d['gz_score_staic_l'][1:]))
    print('*' * 100)

    return gz_aver_score


orig = {
    ' *\( *': '（',
    ' +.': '.',
    ' +,': ',',
    ' *\) *': '）',
    ' *。 *': '。',
    ' *， *': '，',
}

full_half = {
    '，': ',',
    '；': ';',
    '！': '!',
    '：': ':',
    '？': '?',
}


def replace_symbol_ada(string, subject):
    # 句号、句点
    # print("before sym process:", string)
    string = re.sub(r" +", " ", string)
    if subject in ["语文", "政治", "历史", "地理", "生物", "化学"]:
        # 统一中文句号；除小数点外，所有半角句点都被替换成句号；数字和字母后面跟着的句号、句点不变
        string = re.sub(r"([^0-9a-zA-Z]) *[．。] *", r"\1。", string)  # 统一中文句号。。
        string = re.sub(r"([^0-9])\.([^0-9])", r"\1．\2", string)  # 除小数点外，所有半角句点都被替换成全角句点
        # string = re.sub(r"\.([^0-9])", r"。\1", string)  # 除小数点外，所有半角句点都被替换成句号
        string = re.sub(r"(^[0-9A-H]+) *[．。] *", r"\1．", string)  # 句首题号后面跟着的处理为全角句点
    elif subject in ["数学", "物理"]:
        string = re.sub(r"。", r"．", string)  # 所有句号都替换为全角句点
        string = re.sub(r"([^0-9])\.", r"\1．", string)  # 除小数点外，所有半角句点都被替换成全角句点
        string = re.sub(r"\.([^0-9])", r"．\1", string)  # 除小数点外，所有半角句点都被替换成全角句点
    elif subject == "英语":
        # 句号/句点前如果为英文、数字，统一使用英文句点，如果为中文，统一使用句号
        string = re.sub(r"([a-zA-Z0-9])[．。]", r"\1.", string)
        string = re.sub(r"([\u4e00-\u9fa5])[．\.]", r"\1。", string)
        string = re.sub(r' +\.', r".", string)
    else:
        print("[Warning]Unprocessed period. subject:", subject)

    # 全部采用全角中文括号，并清除括号前后的空格
    string = re.sub(r" *[\(（] *", r"（", string)
    string = re.sub(r" *[\)）] *", r"）", string)

    if subject == "英语":
        # 中文前后要使用全角符号，全角符号前后不能有空格；其余全部半角符号
        for sym in full_half:
            string = re.sub(r" *[{}{}]".format(sym, full_half[sym]), r"{}".format(full_half[sym]), string)
            string = re.sub(r"([\u4e00-\u9fa5]) *[{}] *".format(full_half[sym]), r"\1{}".format(sym), string)
            string = re.sub(r" *[{}] *([\u4e00-\u9fa5])".format(full_half[sym]), r"{}\1".format(sym), string)

        # 空白括号（）内无空格
        string = re.sub(r" *（ *） *", r"（）", string)
        # string = eng_underline_postprocess(string)

        # 处理完形填空的问题
        string = re.sub(r" *_+ *([0-9]+) *_+", r"（\1）________", string)
        # string = re.sub(r"([^_ ]) *([0-9]+) *_+", r"\1（\2）________", string)
        string = re.sub(r"([0-9]+) *_+", r"（\1）________", string)
        string = re.sub(r" *[^_]_{1,7} *([0-9]+)", r"（\1）________", string)

        # 下划线后如果是标点，不空格，如果不是标点，空一格
        string = re.sub(r"([0-9a-zA-Z\.\,\!\:\?]) *(_+) +(\W)", r"\1 \2\3", string)
        string = re.sub(r"([0-9a-zA-Z\.\,\!\:\?]) *(_+)([^\W_])", r"\1 \2 \3", string)

    else:
        for sym in full_half:
            string = re.sub(r" *[{}{}] *".format(sym, full_half[sym]), r"{}".format(sym), string)
        string = re.sub(r" *（ *） *", r"（        ）", string)  # （        ）用中文形式，中间空8个字符

    if subject == "生物":
        string = re.sub(r" *【 *", r"[", string)
        string = re.sub(r" *】 *", r"]", string)
        string = re.sub(r" *\[ *\] *", r"[     ]", string)

        string = re.sub(r" *([:：∶]) *", r"\1", string)
    # print("after sym process:", string)
    return string


def underline_bracket_postprocess(text, subject):
    # print(text)
    underline = "_" * 15 if "语文" == subject else "_" * 8
    spaces = ' ' * 8
    if subject == '英语':
        spaces = ''
        text = re.sub(r"(_+) +(\W)", r"\1\2", text)
        text = re.sub(r"(_+)([^\W_])", r"\1 \2", text)
    if subject == "生物":
        text = re.sub(r"【(.*?)】", "[\1]", text)
        spaces = ' ' * 5
    text = re.sub(r"([\(\[【]) *?([\)\]】])", r"\1{}\2".format(spaces), text)
    text = re.sub(r"^_+|_+$|__+", underline, text)
    # print(text)
    return text


def text_pre_process(text, subject=None):
    text = re.sub(r"(<br>)+", " " if subject == '英语' else '', text)
    text = re.sub(r"(<br/>)+", " " if subject == '英语' else '', text)

    # 去掉公式
    if subject in ['数学', '物理', '生物', '化学']:
        text = re.sub(r'\${.+?}\$', r'', text)

    # 去掉题号和选项号
    text = re.sub(r"(^[0-9A-H]+) *[．\.、] *", r"", text)
    text = re.sub(r"^（([0-9]+)）", r"\1", text)
    # text = re.sub(r"（([0-9]+)）", r"\1", text)
    # text = re.sub(r"\([0-9]+\)", r"", text)
    text = re.sub(r"[\(（]\d+?分[\)）]|([(（]本题.*?\d.*?分.*?[）)])", r"", text)
    text = re.sub(r"【题文】", r"", text)
    text = re.sub(r"^[一|二|三|四|五|六|七|八|九]+[|.|．｜、]", r"", text)
    text = re.sub(r"^[(（][一二三四五六七八九][)）].*?(选词填空|综合填空)", r"", text)
    text = re.sub(r"\\rm", r"", text)
    text = re.sub(r'第.+节.*（.*\d分）', r'', text)

    # 完形填空
    text = re.sub(r' *（ *\d+ *） *_*', r'（）________', text)

    text = re.sub(r'.*书面表达', r'', text)
    text = re.sub(r'.*作文.{0,3}\d+分', r'', text)
    text = re.sub(r'（.*满分\d+分）', r'', text)

    text = re.sub(r' *请.{0,2}把.{0,2}答.{0,2}案.{0,2}写.{0,2}到.{0,2}答.{0,2}题.{0,2}卡.{0,2}指.{0,2}定.{0,2}的.{0,2}位.{0,2}置.*',
                  r'', text)

    text = re.sub(' +', '' if subject != '英语' else ' ', text)

    return text


def deal_score(text, text_recg, subject, delsym=False, debug_img_path=None,
               debug_dir=None, vis_thresh=20, ignore_err=False):
    text = replace_symbol_ada(text, subject)
    text = underline_bracket_postprocess(text, subject)

    if delsym:
        text = trans_chi(text, subject)
        text_recg = trans_chi(text_recg, subject)
    # else:
    # text_recg = replace_symbol_ada(text_recg, subject)

    gz_score = Levenshtein.distance(text.strip(), text_recg.strip())
    if max(len(text.strip()), len(text_recg.strip())) != 0:
        norm_score = gz_score / max(len(text.strip()), len(text_recg.strip()))
    else:
        norm_score = 0

    if debug_img_path is not None and gz_score > vis_thresh:
        vis_img(debug_img_path, text, text_recg, debug_dir, gz_score)

    if ignore_err and gz_score > vis_thresh:
        return -1, None, None

    return gz_score, norm_score, text.strip(), text_recg.strip()


subject_name_en2cn = {'biology': '生物', 'chemistry': '化学', 'geography': '地理', 'history': '历史', 'math': '数学',
                      'physics': '物理', 'politics': '政治', 'chinese': '语文', 'english': '英语', 'barcode': 'barcode'}

'''

python ocr_assess_2gz.py  --test_data_dir=test_data_191115_res_229 --delsym=0 

'''
if __name__ == '__main__':
    root_dir = '/media/chen/mcm/ctpn_test_2019_01_23/ocr_KBM_test/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard_data_dir', type=str, default='test_data_200224',
                        help='path to standard_data_dir')
    parser.add_argument('--test_data_dir', type=str, default='test_data_191115_short_1204',
                        help='path to latest pred vis dir')
    parser.add_argument('--standard_label_suffix', type=str, default='_text.txt',
                        help='standard label suffix')
    parser.add_argument('--test_label_suffix', type=str, default='.txt',
                        help='test_label_suffix')
    parser.add_argument('--delsym', type=int, default=1,
                        help='whether del symbol,default is no')
    parser.add_argument('--save_fail_files', type=int, default=0,
                        help='whether save fail files')
    parser.add_argument('--debug', type=int, default=0,
                        help='output debug info')
    parser.add_argument('--debug_dir', type=str, default='',
                        help='output debug info')
    parser.add_argument('--ignore_thresh', type=int, default=20,
                        help='> thresh ignored')
    parser.add_argument('--ignore_cal', type=int, default=0,
                        help='> thresh ignored to cal edit distance')
    opt = parser.parse_args()
    img_ext_list = ['jpg', 'png', 'bmp', 'jpeg']

    # 'test_data_191115'
    # 'test_data_191115_res_229'
    # '_text.txt'
    # '_gz_text.txt'
    standard_data_dir = root_dir + opt.standard_data_dir
    test_data_dir = root_dir + opt.test_data_dir
    standard_label_suffix = opt.standard_label_suffix
    test_label_suffix = opt.test_label_suffix

    delsym = opt.delsym
    debug = opt.debug
    debug_dir = test_data_dir + opt.debug_dir
    if not os.path.exists(debug_dir) and debug:
        print('create debug dir: ', debug_dir)
        os.mkdir(debug_dir)

    subjects = []
    gz_avg_edit_distance = []
    fail_exist_file = []
    e_count = 0
    t_count = 0
    for en_subject in os.listdir(standard_data_dir):
        subject = subject_name_en2cn.get(en_subject.split('_', 1)[1])
        score_d = create_score_d()
        staic_d = create_staic_d()

        sub_dir_path = os.path.join(standard_data_dir, en_subject)
        stand_label_list = glob(os.path.join(sub_dir_path, '*' + '_text.txt'))

        for path in stand_label_list:
            test_label = os.path.join(test_data_dir, en_subject, os.path.basename(path).replace(standard_label_suffix,
                                                                                                test_label_suffix))

            if os.path.exists(test_label):
                stand_line = _read_content(path)
                label_line = _read_content(test_label, 'gz', subject)

                debug_img = None
                if opt.debug:
                    debug_img_paths = [test_label.replace(test_label_suffix, '.' + ext) for ext in img_ext_list]
                    debug_img_paths = [path for path in debug_img_paths if os.path.exists(path)]

                    debug_img = debug_img_paths[0]
                    if len(debug_img_paths) != 1:
                        print('debug img path not only:', debug_img_paths)

                n_stand_line = text_pre_process(stand_line, subject)
                n_label_line = text_pre_process(label_line, subject)

                gz_score, norm_score, text_stand, text_label = deal_score(n_stand_line, n_label_line,
                                                                          subject,
                                                                          delsym=delsym, debug_img_path=debug_img,
                                                                          debug_dir=debug_dir,
                                                                          vis_thresh=opt.ignore_thresh,
                                                                          ignore_err=opt.ignore_cal)

                if gz_score == -1 and opt.ignore_cal:
                    continue
                # if gz_score > 20 and subject == '英语':
                #     print(path.replace(standard_label_suffix, ".png"))
                #     # print(stand_line)
                #     print(text_stand)
                #     print(text_label)
                #     img_name = os.path.split(path.replace(standard_label_suffix, ".png"))[-1]
                #     shutil.copyfile(path.replace(standard_label_suffix, ".png"), "/media/chen/wt/tools/" + img_name)
                #     # print(text_label)
                #     # try:
                #     #     os.remove(path.replace(standard_label_suffix, ".png"))
                #     # except:
                #     #     pass
                #     # try:
                #     #     os.remove(path.replace(standard_label_suffix, ".txt"))
                #     # except:
                #     #     pass
                #     # try:
                #     #     os.remove(path)
                #     # except:
                #     #     pass
                #
                #     # e_count += 1
                #
                #     text_pre_process(stand_line, subject)
                #     text_pre_process(label_line, subject)

                gz_index = get_index(gz_score)

                if gz_score == 0:
                    e_count += 1

                staic_d['gz_score_staic_l'][gz_index] += 1
                score_d['gz_score_l'].append(gz_score)

            else:
                fail_exist_file.append(test_label)
            t_count += 1

        insert_staic(staic_d)
        gz_avg_score = print_static_info(score_d, staic_d, en_subject)
        # 科目
        period, subject_name = en_subject.split('.')[0].split('_')
        subjects.append(en_subject)

        # 广州平均编辑距离
        gz_avg_edit_distance.append(gz_avg_score)
    print(e_count, t_count, "%.2f" % (e_count / t_count))
    subjects.append('总平均')
    gz_avg_edit_distance.append(sum(gz_avg_edit_distance) / len(gz_avg_edit_distance))
    cur_mon = get_cur_mon()
    df = pd.DataFrame({'科目': subjects, '{}月平均编辑距离(广州)'.format(cur_mon): gz_avg_edit_distance})

    save_log_path = '{}_mon_edit_distance_log_{}_nodelsym.csv'
    if delsym:
        save_log_path = '{}_mon_edit_distance_log_{}_delsym.csv'
    save_log_path = save_log_path.format(cur_mon, os.path.basename(test_data_dir))
    df.to_csv(save_log_path, encoding='utf_8_sig')

    print('files not exists in pred data dir:', fail_exist_file)
    print(len(fail_exist_file))
    print('res save to {}'.format(save_log_path))
    if debug:
        print('debug dir: ', debug_dir)
    # 为识别失败的文件建立相同结构的存档方便针对性重测
    if len(fail_exist_file) and opt.save_fail_files:

        test_data_lack_dir = os.path.join(os.path.dirname(test_data_dir), os.path.basename(test_data_dir) + '_lack')
        if not os.path.exists(test_data_lack_dir):
            print('creating {}'.format(test_data_lack_dir))
            os.mkdir(test_data_lack_dir)

        for fail_file in fail_exist_file:
            basename = os.path.basename(fail_file).replace(test_label_suffix, '')
            lack_sub_dir_name = os.path.basename(os.path.dirname(fail_file))
            lack_sub_dir = os.path.join(test_data_lack_dir, lack_sub_dir_name)
            if not os.path.exists(lack_sub_dir):
                # print('creating {}'.format(lack_sub_dir))
                os.mkdir(lack_sub_dir)

            fail_file_in_standdir = glob(os.path.join(standard_data_dir, lack_sub_dir_name, '{}.*'.format(basename)))
            fail_file_in_standdir = [path for path in fail_file_in_standdir if
                                     os.path.basename(path).split('.')[-1].lower() in img_ext_list]

            for ff_path in fail_file_in_standdir:
                dst_path = os.path.join(lack_sub_dir, os.path.basename(ff_path))
                shutil.copy(ff_path, dst_path)
                # print('copied {}'.format(dst_path))
