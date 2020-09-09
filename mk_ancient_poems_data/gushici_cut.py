import os
from PIL import Image, ImageDraw
import json
import glob
import cv2
import tqdm
import shutil


def read_json(json_path):
    """
    Interpret json string.
    :param json_path: path to json file.
    :return: dict type interpreted from json file.
    """
    with open(json_path, "r") as fr:
        ret_dict = json.loads(fr.read())
    return ret_dict


def get_stus(jsondir):
    stus = []
    jsonfiles = glob.glob(os.path.join(jsondir, '*.json'))
    for jsonfile in jsonfiles:
        if 'report' in jsonfile:
            json = read_json(jsonfile)
            if not json:
                continue
            stus.extend(json['[stus]'])
    return stus


def get_img_dict(imgpaths):
    img_dict = {}
    for imgpath in imgpaths:
        basename = os.path.basename(imgpath)
        if basename.split('_')[3].split('.')[0] not in img_dict:
            img_dict[basename.split('_')[3].split('.')[0]] = imgpath
    return img_dict


def get_id_dict(rec_txtpaths):
    """
    获取id跟模板图片对应的txt文件名的字典
    :param rec_imgpaths:
    :return:
    """
    id_dict = {}
    for rec_txtpath in rec_txtpaths:
        if not 'info.txt' in rec_txtpath:
            basename = os.path.basename(rec_txtpath)
            if basename.split('_')[0] not in id_dict:
                id_dict[basename.split('_')[0]] = rec_txtpath
    return id_dict


def write_content_txt(content, contenttxtpath):
    contentfile = open(contenttxtpath, 'w')
    for line in content:
        contentfile.write(line + '\n')
    contentfile.close()


def get_recs(txtpath):
    """
    获取诗句的方框位置及标注
    :param txtpath:
    :return:
    """
    recs = open(txtpath, 'r', encoding='utf-8').read().strip('\n').split('\n')
    boxs = []
    for rec in recs:
        rec_split = rec.split(',', 4)
        if len(rec_split) > 4:
            x0, y0, x1, y1, label = rec_split[:5]
            if not label.strip() == 'n':
                boxs.append([int(x0), int(y0), int(x1), int(y1), label])
    return boxs


def get_blockid_score(txtpath):
    lines = open(txtpath, 'r', encoding='utf-8').read().strip('\n').split('\n')
    blockid = ''
    score = ''
    if len(lines) > 0:
        blockid = lines[0].split(' ', 1)[0]
        score = lines[0].split(' ', 1)[1]
    return blockid, score


def get_full_score_kaohao(jsondir, contentpath):
    lines = open(contentpath, 'r').read().strip('\n').split('\n')
    blockid = ''
    fullscore = ''

    if len(lines) > 0:
        blockid = lines[0].split(' ')[0]
        fullscore = lines[0].split(' ')[1]
    stus = get_stus(jsondir)
    block_num = ''
    blocks = stus[0]['[blocks]']
    full_score_kaohao = ''
    full_score_xuehao = ''
    for i in range(len(blocks)):
        if blocks[i]['id'] == int(blockid):
            block_num = i
    if not block_num == '':
        for stu in tqdm.tqdm(stus):
            name = stu['name']
            if not name:  # 跳过名字为空的情况？？？？
                continue
            kaohao = stu['kaohao']

            if not kaohao:  # 跳过交白卷的情况!!!
                continue
            xuehao = stu['xuehao']
            if not xuehao:
                continue
            try:
                score = stu['[blocks]'][block_num]['score']
                if score == int(fullscore):
                    full_score_kaohao = kaohao
                    full_score_xuehao = xuehao
                    break
            except:
                print('学生不包含该block')

    return full_score_kaohao, full_score_xuehao


def get_full_score_img(data_path, block_rec_path, gushici_rec_path, dealdata_path):
    """

    :param data_path: 原始数据文件夹
    :param block_rec_path: 可视化块位置id，满分的文件夹
    :param gushici_rec_path: 放置一张满分图片坐模板的文件夹
    :param dealdata_path: 已处理的数据文件夹
    :return:
    """
    iddeal_dict = {}
    rec_pngs = glob.glob(os.path.join(gushici_rec_path, '*.png'))
    for rec_png in rec_pngs:
        iddeal_dict[os.path.basename(rec_png).split('_')[1]] = 0
    dealids = os.listdir(dealdata_path)
    for dealid in dealids:
        iddeal_dict[dealid] = 0
    idfolders = os.listdir(data_path)
    block_rec_txts = glob.glob(os.path.join(block_rec_path, '*_content.txt'))
    block_rec_dict = get_id_dict(block_rec_txts)

    for idfolder in tqdm.tqdm(idfolders):
        if os.path.isdir(os.path.join(data_path, idfolder)):
            try:
                imgpaths = glob.glob(os.path.join(data_path, idfolder, 'src', '*.png'))
                kaoshiid = ''
                schoolid = ''
                if len(imgpaths) > 0:
                    kaoshiid = os.path.basename(imgpaths[0]).split('_')[1]
                    schoolid = os.path.basename(imgpaths[0]).split('_')[2]
                imgdict = get_img_dict(imgpaths)
                jsondir = os.path.join(data_path, idfolder, 'info')  # json文件
                if idfolder not in block_rec_dict:
                    continue
                if idfolder in iddeal_dict:
                    continue
                contentpath = block_rec_dict[idfolder]
                txtpath = contentpath.replace('_content.txt', '.txt')
                if not os.path.exists(txtpath):
                    continue
                boxs = get_recs(txtpath)
                if len(boxs) == 0:
                    continue
                full_score_kaohao, full_score_xuehao = get_full_score_kaohao(jsondir, contentpath)
                itemkey = ''
                if not full_score_kaohao == '' and not full_score_xuehao == '':
                    if full_score_kaohao in imgdict:
                        itemkey = full_score_kaohao
                    else:
                        itemkey = full_score_xuehao
                    image = cv2.imread(imgdict[itemkey])
                    for box in boxs:
                        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(gushici_rec_path, os.path.basename(imgdict[itemkey])), image)
                    # shutil.copy(imgdict[itemkey], gushici_rec_path)
                    # 复制content文件，并改名字
                    shutil.copy(contentpath, gushici_rec_path)
                    os.rename(os.path.join(gushici_rec_path, os.path.basename(contentpath)),
                              os.path.join(gushici_rec_path,
                                           os.path.basename(
                                               imgdict[
                                                   itemkey]).split(
                                               '.')[
                                               0] + '_content.txt'))
            except:
                pass


def get_mask_img(data_path, rec_path, dealdata_path):
    """
    从每一场考试中拷贝出一张图片，作为模板，并把块可视化出来
    :param data_path:下载数据的位置
    :param rec_path:放置模板图片的位置
    :param dealdata_path: 已处理过id的id列表文件，没有的话，直接给空txt文件也可以
    :return:
    """
    # 获取已经选取的id
    iddeal_dict = {}  # iddeal_dict 为已处理过的id
    rec_pngs = glob.glob(os.path.join(rec_path, '*.png'))
    for rec_png in rec_pngs:
        iddeal_dict[os.path.basename(rec_png).split('_')[1]] = 0
    # 获取其他路径下已取mask_img的id
    deal_data = open(dealdata_path, 'r').read().strip()
    lines = deal_data.split('\n') if deal_data != '' else []
    for line in lines:
        ids = os.listdir(line)
        for id in ids:
            if 'png' not in id:
                continue
            if id.split('_')[1] not in iddeal_dict:
                iddeal_dict[id.split('_')[1]] = 0
    idfolders = os.listdir(data_path)
    for idfoler in tqdm.tqdm(idfolders):
        if os.path.isdir(os.path.join(data_path, idfoler)):
            if idfoler not in iddeal_dict:
                imgpaths = glob.glob(os.path.join(data_path, idfoler, 'src', '*.png'))
                if len(imgpaths) >= 100:
                    try:
                        json = read_json(os.path.join(data_path, idfoler, 'info', 'block.json'))
                        img = cv2.imread(imgpaths[0])
                        blocks_info = json["sub_info"]
                        # blockscore_infos = json['[blocks]']
                        # if not blockscore_infos:
                        #     continue
                        # score_dict = {}
                        # for blockscroe_info in blockscore_infos:
                        #     points = blockscroe_info['[points]']
                        #     id = blockscroe_info['blockid']
                        #     fullscore = 0
                        #     for point in points:
                        #         fullscore += point['score']
                        #     score_dict[id] = fullscore
                        if not blocks_info:
                            continue
                        # 将块id信息记录下来，保存成txt文件
                        contenttxt = []
                        for block_info in blocks_info:
                            blockid = block_info.get("blockid")
                            block_pos = block_info.get("block_pos")
                            for b_pos in block_pos:
                                page_no, x, y, w, h = b_pos[:5]
                                if page_no != 0:
                                    continue
                                # contenttxt.append(str(blockid) + ' ' + str(score_dict[blockid]) )
                                contenttxt.append('{} {} {} {} {}'.format(blockid, x, y, x + w, y + h))
                                cv2.putText(img, str(blockid), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.imwrite(os.path.join(rec_path, os.path.basename(imgpaths[0])), img)
                        write_content_txt(contenttxt,
                                          os.path.join(rec_path, os.path.basename(imgpaths[0])).replace('.png',
                                                                                                        '_content.txt'))
                    except:
                        pass


def get_gushici_cut(stus, png_dict, boxs, blockid, fullscore, labelpath, des_path):
    """
        :param stus:考生信息列表
        :param png_dict: 考号跟图片路径对应的字典
        :param boxs: 方框坐标及label
        :param labelpath:
        :return:
        """
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    labelfile = open(labelpath, 'w', encoding='utf-8')
    for i in range(len(boxs)):
        labelfile.write(str(i) + ' ' + str(boxs[i][4]) + '\n')
        if not os.path.exists(os.path.join(des_path, str(i))):
            os.mkdir(os.path.join(des_path, str(i)))
    labelfile.close()
    block_num = ''
    blocks = stus[0]['[blocks]']
    for i in range(len(blocks)):
        if blocks[i]['id'] == int(blockid):
            block_num = i
    if not block_num == '':
        try:
            for stu in tqdm.tqdm(stus):
                name = stu['name']
                if not name:  # 跳过名字为空的情况？？？？
                    continue
                kaohao = stu['kaohao']

                if not kaohao:  # 跳过交白卷的情况!!!
                    continue
                xuehao = stu['xuehao']
                if not xuehao:
                    continue
                score = stu['[blocks]'][block_num]['score']
                if score == int(fullscore):
                    if kaohao in png_dict:
                        pass
                    if xuehao in png_dict:
                        kaohao = xuehao
                    img = Image.open(png_dict[kaohao])
                    for i in range(len(boxs)):
                        imgcrop = img.crop((boxs[i][0], boxs[i][1], boxs[i][2], boxs[i][3]))
                        imgcrop.save(os.path.join(des_path, str(i),
                                                  os.path.basename(png_dict[kaohao]).split('.')[0]
                                                  + '_' + str(i) + '.png'))
        except:
            print("该学生分数有问题")


def gushici_cut(data_path, rec_path, des_path):
    """
        get the chinese name from paperdata
        :param rec_path: 名字位置方框数据文件夹
        :param data_path: 试卷数据路径文件夹
        :param des_path: 数据保存文件夹
        :return:
        """
    # 获取所有的记录方框数据的txt文件路径
    txtpaths = [i.replace('.png', '.txt') for i in glob.glob(os.path.join(rec_path, '*.png'))]
    for txtpath in txtpaths:
        if not txtpath == os.path.join(rec_path, 'info.txt') and os.path.exists(txtpath):
            boxs = get_recs(txtpath)
            content_path = os.path.join(rec_path, os.path.basename(txtpath).split('.')[0] + '_content.txt')
            blockid, score = get_blockid_score(content_path)
            kaoshiid = os.path.basename(txtpath).split('_')[1]
            schoolid = os.path.basename(txtpath).split('_')[2]
            # 判断是否已经裁剪过了，裁剪过了就跳过
            if os.path.exists(os.path.join(des_path, kaoshiid)):
                print(kaoshiid + '已裁剪过')
                continue
            if os.path.exists(os.path.join(des_path, kaoshiid)):
                continue
            # 判断info文件夹是否存在
            if not os.path.exists(os.path.join(data_path, kaoshiid, 'info')):
                continue
            # 判断src文件夹是否存在
            if not os.path.exists(os.path.join(data_path, kaoshiid, 'src')):
                continue
            # 将考号跟图片名字对应起来的字典
            img_dict = {}
            pngs = glob.glob(os.path.join(data_path, kaoshiid, 'src', '*.png'))
            for png in pngs:
                img_dict[os.path.basename(png).split('_')[3].split('.')[0]] = png
            jsondir = os.path.join(data_path, kaoshiid, 'info')
            stus = get_stus(jsondir)
            print(kaoshiid + '\n')
            # print(len(stus))
            labelpath = os.path.join(des_path, kaoshiid, 'label.txt')
            get_gushici_cut(stus, img_dict, boxs, blockid, score, labelpath, os.path.join(des_path, kaoshiid))


if __name__ == '__main__':
    runtype = 2
    if runtype == 0:  # 可视化块id、保存块id信息
        blockrecdirfile = "/media/chen/mcm/data/chinese_paper/block_rec_dir.txt"
        get_mask_img(r"/media/chen/mcm/data/chinese_paper/data", r"/media/chen/mcm/data/chinese_paper/block_rec_5",
                     blockrecdirfile)
    elif runtype == 1:  # 获取一张满分的试卷作为模板
        get_full_score_img(r"/media/chen/mcm/data/chinese_paper/data",
                           r"/media/chen/mcm/data/chinese_paper/block_rec_5",
                           r"/media/chen/mcm/data/chinese_paper/gushici_rec_5",
                           r"/media/chen/mcm/data/chinese_paper/poem_test")
    elif runtype == 2:  # 进行裁剪
        gushici_cut(r"/media/chen/mcm/data/chinese_paper/data", r"/media/chen/mcm/data/chinese_paper/gushici_rec_5",
                    r"/home/chen/project/ocr/hcn.crnn/data/hand/cn/gushici/test_data")
    elif runtype == 3:  # 测试
        get_stus(r"/media/chen/mcm/data/chinese_paper/data/1002598/info")
