import json
from collections import Counter
import re
import os
import cv2
import tarfile
import tqdm
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import time
import shutil
import glob

json_file = r'/media/chen/wt/data/hw_poem_labels/corpus/ancient_poems.json'
label_file = r'/media/chen/wt/data/hw_poem_labels/hw_labels_all_train.txt'
label_vocab_file = r'/media/chen/wt/data/hw_poem_labels/hw_labels_all_train_vocab.txt'
label_poems_file = r'/media/chen/wt/data/hw_poem_labels/poem_labels_clean.txt'
junior_poems_file = r'/media/chen/wt/data/hw_poem_labels/corpus/junior_ancient_poems.txt'
senior_poems_file = r'/media/chen/wt/data/hw_poem_labels/corpus/senior_ancient_poems.txt'

biaodian = r"\,\.\!\?\:\'\"《》。，‘“’”‘’！？\[\]【】\{\}\-—_；\;、 ：（）\(\)　．·…"


def corpus_parse():
    statistic = {'初中': set(), '高中': set()}
    poems = {}
    poem_sentences = set()
    poem_vocab = set()
    label_word_freq_counter = Counter()
    poem_len = {}
    label_poems = set()

    dagang_from_weifang = [junior_poems_file, senior_poems_file]
    dagang_poem_list = []
    dagang_vocab = set()
    for file in dagang_from_weifang:
        with open(file) as f:
            new_lines = [line for line in f if not re.search(r'[七八九](年级)?[上下]|[七八九]年级|第.单元|必修|选修|古诗词|课外|课内', line)]
            long_line = ''.join(new_lines)
            long_line = re.sub(r'——《.+?》', '', long_line)
            poem_list = re.sub(r'(《.+?》.*?\n)', '', long_line)
            # poem_list = [re.sub('[ \n]+', '', poem.strip()) for poem in poem_list if len(poem.strip()) != 0]
            poem_list = re.split(r'\n', poem_list)
            poem_list = [poem.strip() for poem in poem_list if len(poem.strip()) != 0]
            [dagang_poem_list.extend(re.split(r'[%s]' % biaodian, poem)) for poem in poem_list]
            print(new_lines[0])
    dagang_poem_list = [poem for poem in dagang_poem_list if len(poem) != 0]
    dagang_vocab = dagang_vocab | set(''.join([re.sub(r'[%s]+' % biaodian, '', poem) for poem in dagang_poem_list]))

    with open(json_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            poem_attr = json.loads(line)
            poem_content = re.sub(r'<br/>|[\(（].*[\)）]|\n|[　  ]+|\\u2002', '', poem_attr['content'])
            statistic[poem_attr['period']].add(poem_attr['title'])
            if poem_attr['title'] not in poems or len(poems[poem_attr['title']]) < len(poem_content):
                poems[poem_attr['title']] = poem_content
                # poem_len[poem_attr['title']] = len(poem_content)
            sentences = re.split(r'[%s]+' % biaodian, poem_content.strip())
            poem_vocab = poem_vocab | set(''.join(sentences))
            poem_sentences = poem_sentences | set(sentences)

        for period in statistic:
            print(period, len(statistic[period]))
    print("网络语料字表长度：", len(poem_vocab))

    with open(label_vocab_file) as f:
        label_vocab = [c for c in f.readline()]
    print('label字典长度', len(label_vocab))

    word_no_see = set(poem_vocab) - set(label_vocab)
    print('word_no_see:\n', ''.join(list(word_no_see)))
    print('len of word_no_see:', len(word_no_see))

    dagang_word_no_see = set(dagang_vocab) - set(label_vocab)
    print('dagang word_no_see:\n', ''.join(list(dagang_word_no_see)))
    print('len of dagang word_no_see:', len(dagang_word_no_see))

    total_word_no_see = dagang_word_no_see | word_no_see
    print('total word no see:\n', ''.join(list(total_word_no_see)))
    print('len of total word no see:\n', len(total_word_no_see))

    dagang_poem_no_see = []
    for poem in dagang_poem_list:
        poem_word = set(re.sub(r'[%s]+' % biaodian, '', poem))
        if len(poem_word & dagang_word_no_see) != 0:
            dagang_poem_no_see.append(poem)
    print('dagang poem sentences no_see:\n', dagang_poem_no_see)
    print('len of poem_no_see:', len(dagang_poem_no_see))

    poem_sentence_no_see = []
    for sentence in poem_sentences:
        poem_sentence_word = set(list(sentence))
        if len(poem_sentence_word & word_no_see) != 0:
            poem_sentence_no_see.append(sentence)
    print('json poem_sentence_no_see:\n', poem_sentence_no_see)
    print('len of poem_sentence:', len(poem_sentences))
    print('len of poem_sentence_no_see:', len(poem_sentence_no_see))

    total_poem_no_see = set(poem_sentence_no_see) | set(dagang_poem_no_see)
    print('total poem no see:\n', total_poem_no_see)
    print('total poem no see count:', len(total_poem_no_see))

    print("total word count:", len(''.join(poem_sentence_no_see)))

    total_poem_sentences = [poem for poem in poem_sentences if len(poem) > 1]
    total_poem_sentences.extend([poem for poem in dagang_poem_list if len(poem) > 1])
    total_poem_sentences = set(total_poem_sentences)
    with open('/media/chen/wt/ocr_hw_poem/service/test/corpus/poem_sentences.txt', 'w') as f:
        f.writelines([poem + '\n' for poem in total_poem_sentences])

    with open(label_poems_file) as f:
        label_poems = [line.split(' ', 1)[1].strip() for line in f.readlines()]
        label_poem_vocab = set(''.join([poem for poem in label_poems]))

    word_no_need = set(label_vocab) - poem_vocab - label_poem_vocab

    print('word_no_need:\n', ''.join(list(word_no_need)))
    print('len of word_no_need:', len(word_no_need))

    word_not_in_corpus = label_poem_vocab - poem_vocab

    print('words in train poem but not in corpus:\n', ''.join(list(word_not_in_corpus)))
    print('len of words in train poem but not in corpus:', len(word_not_in_corpus))

    label_poems = set(label_poems)
    for poem in label_poems:
        if len(set(poem) & word_not_in_corpus) != 0:
            print(poem, '---', set(poem) & word_not_in_corpus)

    print('Done')


def count_poems(paper_dir):
    txt_file = os.path.join(paper_dir, 'hard_poems.txt')
    poems_counter = Counter()
    with open(txt_file) as f:
        poems = [line.split(' ', 1)[-1].strip() for line in f.readlines()]
        for poem in poems:
            poems_counter[poem] += 1
    return poems_counter


def tar_labels_to_file(lines, dst_tar_file, arc_prefix):
    tar = tarfile.open(dst_tar_file, "w:gz")
    for line in lines:
        img_path, gt = line.split(' ', 1)
        try:
            w, h = Image.open(img_path).size
        except:
            print("Read img {} error".format(img_path))
            continue
        img_txt = re.sub(r'(png|jpg|PNG|JPEG|JPG)$', r'txt', img_path)
        if not img_txt.endswith('txt'):
            continue
        with open(img_txt, "w") as f:
            f.write('0, 0, {}, {}, '.format(w, h) + gt)
        tar.add(img_path, arcname=os.path.join(arc_prefix, os.path.split(img_path)[-1]))
        tar.add(img_txt, arcname=os.path.join(arc_prefix, os.path.split(img_txt)[-1]))
    tar.close()


def tar_label_data():
    papers_root_dir = r"/media/chen/wt/ocr_hw_poem/service/test/papers"
    id_record_file = r"/media/chen/wt/ocr_hw_poem/service/test/processed.txt"
    papers_dst_tar_dir0 = r"/media/chen/wt/ocr_hw_poem/service/test/tar_dir"
    papers_dst_tar_dir1 = r"/media/chen/wt/ocr_hw_poem/service/test/tar_dir_1"
    papers_dst_tar_dir_prefix = r"/media/chen/wt/ocr_hw_poem/service/test/tar_dir_"
    valid_finished_dirs = []

    id_tared = set([id_dir.split('.')[0] for id_dir in os.listdir(papers_dst_tar_dir0)])

    with open(papers_dst_tar_dir1 + '/label.txt') as f:
        sampled_files = set([line.split(' ', 1)[0] for line in f.readlines()])

    with open(id_record_file) as f:
        id_finished_dirs = set([line.strip() for line in f.readlines()])

    for id_dir in os.listdir(papers_root_dir):
        # id_dir = '1278846'
        if id_dir not in id_finished_dirs:
            continue
        if id_dir in id_tared:
            continue
        valid_finished_dirs.append(id_dir)
        # poems_counter = poems_counter + count_poems(os.path.join(papers_root_dir, id_dir))

    print("Count poems...")
    poem_dirs_map = {}
    label_lines = []
    for id_dir in tqdm.tqdm(valid_finished_dirs):
        id_label_path = os.path.join(papers_root_dir, id_dir, 'hard_poems.txt')
        with open(id_label_path) as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.split(' ', 1)
                if path in sampled_files:
                    continue
                label_lines.append(line)

    random.shuffle(label_lines)
    for i in range(0, len(label_lines), 50000):
        print("tar files index: %02d...." % (i // 50000))
        papers_dst_tar_dir = papers_dst_tar_dir_prefix + '%02d' % (i // 50000)
        dst_labels = label_lines[i:i + 50000]
        if not os.path.exists(papers_dst_tar_dir):
            os.makedirs(papers_dst_tar_dir)

        with open(os.path.join(papers_dst_tar_dir, 'label.txt'), 'w') as f:
            executor = ThreadPoolExecutor(20)
            all_task = []
            for j in tqdm.tqdm(range(0, len(dst_labels), 1000)):
                dst_tar_file = os.path.join(papers_dst_tar_dir, '%06d' % j + '.tar.gz')
                tile_labels = dst_labels[j: j + 1000]
                f.writelines(tile_labels)
                # tar_labels_to_file(tile_labels, dst_tar_file, '%06d' % j)
                task = executor.submit(tar_labels_to_file, tile_labels, dst_tar_file, '%06d' % j)
                all_task.append(task)
            # wait(all_task, return_when=ALL_COMPLETED)
            task_done = 0
            while task_done < len(all_task):
                task_status = [task.done() for task in all_task]
                task_done = sum(task_status)
                print('done/total: %d/%d, task left: %d' % (task_done, len(all_task), len(all_task) - task_done))
                time.sleep(5)


def mk_data_from_orig_pic(orig_root, dst_root):
    """
    从春娜回收回来的学霸答案制作数据中，按行进行裁剪，得到识别数据
    :param orig_root:
    :param dst_root:
    :return:
    """
    sub_dirs = os.listdir(orig_root)
    all_labels = []
    for i, sub_dir in tqdm.tqdm(enumerate(sub_dirs)):
        txt_files = glob.glob(os.path.join(orig_root, sub_dir) + '/*.txt')
        dst_sub_dir = os.path.join(dst_root, "%06d" % i)
        if not os.path.exists(dst_sub_dir):
            os.makedirs(dst_sub_dir)
        dst_sub_dir_label_file = os.path.join(dst_sub_dir, 'label.txt')
        with open(dst_sub_dir_label_file, 'w') as fw:
            for txt in txt_files:
                with open(txt) as f_txt:
                    lines = f_txt.readlines()
                try:
                    img = cv2.imread(txt.replace('.txt', '.png'))
                except Exception as e:
                    print(e)
                    continue
                file_prefix = os.path.splitext(os.path.split(txt)[-1])[0]
                if len(lines) > 99:
                    print('Warning: lines in one image surpass max count, discard it')
                    continue
                for j, line in enumerate(lines):
                    split_items = line.strip().split(',', 4)
                    x1, y1, x2, y2 = list(map(int, split_items[:4]))
                    if x1 >= x2 - 5 or y1 >= y2 - 5:
                        continue
                    label = split_items[-1].strip()
                    sub_img = img[y1:y2, x1:x2, :]
                    sub_img_file = os.path.join(dst_sub_dir, file_prefix + '_%02d' % j + '.png')
                    cv2.imwrite(sub_img_file, sub_img)
                    poem_label = sub_img_file + ' ' + label + '\n'
                    fw.write(poem_label)
                    all_labels.append(poem_label)
    with open(os.path.join(dst_root, 'all_labels.txt'), 'w') as fw:
        fw.writelines(all_labels)
    random.shuffle(all_labels)
    val_labels = all_labels[:len(all_labels) // 10]
    train_labels = all_labels[len(all_labels) // 10:]
    train_file = os.path.join(dst_root, 'train_labels.txt')
    val_file = os.path.join(dst_root, 'val_labels.txt')
    with open(train_file, 'w') as f:
        f.writelines(train_labels)
    with open(val_file, 'w') as f:
        f.writelines(val_labels)


def tar_hw_general_hard_data():
    label_file = r'/media/chen/wt/data/hw_train_data_hard_sentences.txt'
    dst_root_dir = r'/media/chen/wt/data/hw_train_data_hard/'
    papers_dst_tar_dir_prefix = dst_root_dir + 'tar_file_'
    with open(label_file) as f:
        label_lines = [' '.join(line.split(' ', 2)[::2]) for line in f.readlines()]
    for i in range(0, len(label_lines), 50000):
        print("tar files index: %02d...." % (i // 50000))
        papers_dst_tar_dir = papers_dst_tar_dir_prefix + '%02d' % (i // 50000)
        dst_labels = label_lines[i:i + 50000]
        if not os.path.exists(papers_dst_tar_dir):
            os.makedirs(papers_dst_tar_dir)

        executor = ThreadPoolExecutor(2)
        all_task = []
        for j in tqdm.tqdm(range(0, len(dst_labels), 1000)):
            dst_tar_file = os.path.join(papers_dst_tar_dir, '%06d' % j + '.tar.gz')
            tile_labels = dst_labels[j: j + 1000]
            task = executor.submit(tar_labels_to_file, tile_labels, dst_tar_file, '%06d' % j)
            all_task.append(task)
        # wait(all_task, return_when=ALL_COMPLETED)
        task_done = 0
        while task_done < len(all_task):
            task_status = [task.done() for task in all_task]
            task_done = sum(task_status)
            print('done/total: %d/%d, task left: %d' % (task_done, len(all_task), len(all_task) - task_done))
            time.sleep(5)


def unzip_cleaned_files(root_dir):
    """
    解压缩zip文件
    :param root_dir:
    :return:
    """
    zip_files = [file for file in os.listdir(root_dir) if file.endswith('zip')]
    # rar_files = [file for file in os.listdir(root_dir) if file.endswith('rar')]
    for i, file in enumerate(zip_files):
        os.system('unzip %s -d %s' % (os.path.join(root_dir, file), root_dir + '/' + file[:6]))
    # for i, file in enumerate(rar_files):
    #     os.system('tar -xzvf %s -d %s' % (os.path.join(root_dir, file), root_dir + '/' + file[:29] + '%d' % i))


def postprocess_cleaned_files(root_dir):
    sub_dirs = [file for file in os.listdir(root_dir)]
    for sub_dir in sub_dirs:
        if not os.path.isdir(os.path.join(root_dir, sub_dir)):
            continue
        for root, dirs, files in os.walk(os.path.join(root_dir, sub_dir)):
            img_files = []
            if root.endswith('false'):
                continue
            for file in files:
                file_full_path = os.path.join(root, file)
                if file.endswith('txt') or os.path.isdir(file_full_path) or not os.path.exists(
                        file_full_path[:-3] + 'txt'):
                    continue
                img_files.append(file)
            txt_files = [file[:-3] + 'txt' for file in img_files]
            for file in txt_files + img_files:
                try:
                    shutil.move(os.path.join(root, file), os.path.join(root_dir, sub_dir))
                except shutil.Error:
                    continue
        for file in os.listdir(os.path.join(root_dir, sub_dir)):
            if not os.path.isdir(os.path.join(root_dir, sub_dir, file)):
                continue
            shutil.rmtree(os.path.join(root_dir, sub_dir, file))
    generate_label_file(root_dir)


def generate_label_file(root_dir):
    """
    从labelimg格式（*, *, *, *, gt）的数据生成训练用的数据格式（path gt），并划分测试集和训练集
    :param root_dir:
    :return:
    """
    sub_dirs = os.listdir(root_dir)
    dst_file = os.path.join(root_dir, 'labels.txt')
    with open(dst_file, 'w') as f:
        all_labels = []
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(root_dir, sub_dir)
            txt_files = glob.glob(sub_dir_path + '/*.txt')
            for txt in txt_files:
                with open(txt) as fr:
                    gt = fr.readline().split(',', 4)[-1].strip()
                img_file = os.path.join(sub_dir_path, os.path.splitext(txt)[0] + '.png')
                if not os.path.exists(img_file):
                    img_file = os.path.join(sub_dir_path, os.path.splitext(txt)[0] + '.jpg')
                    if not os.path.exists(img_file):
                        continue
                label = os.path.join(sub_dir_path, img_file) + ' ' + gt + '\n'
                all_labels.append(label)
        f.writelines(all_labels)
    random.shuffle(all_labels)
    val_labels = all_labels[:len(all_labels) // 10]
    train_labels = all_labels[len(all_labels) // 10:]
    train_file = os.path.join(root_dir, 'train_labels.txt')
    val_file = os.path.join(root_dir, 'val_labels.txt')
    with open(train_file, 'w') as f:
        f.writelines(train_labels)
    with open(val_file, 'w') as f:
        f.writelines(val_labels)


def post_process(root_dir):
    sub_dirs = [os.path.join(root_dir, sub_dir) for sub_dir in os.listdir(root_dir) if
                os.path.isdir(os.path.join(root_dir, sub_dir))]
    for sub_dir in sub_dirs:
        txt_files = glob.glob(sub_dir + '/*.txt')
        for txt in txt_files:
            if txt[-8:] == 'info.txt':
                os.remove(txt)
                continue
            img_file = txt[:-3] + 'png'
            if not os.path.exists(img_file):
                img_file = txt[:-3] + 'jpg'

            with open(txt) as f:
                lines = f.readlines()
            if len(lines) == 1:
                # gt = line.split(', ', 4)[-1].strip()
                # all_labels.append('%s %s\n' % (img_file, gt))
                continue
            img = cv2.imread(img_file)
            file_prefix = os.path.splitext(os.path.split(txt)[-1])[0]
            for i, line in enumerate(lines):
                x1, y1, x2, y2, gt = line.split(', ', 4)
                sub_txt_file = os.path.join(sub_dir, file_prefix + "_%d.txt" % i)
                sub_img_file = os.path.join(sub_dir, file_prefix + "_%d.png" % i)
                with open(sub_txt_file, 'w') as f:
                    f.writelines('%s, %s, %s, %s, %s' % (x1, y1, x2, y2, gt))
                cv2.imwrite(sub_img_file, img[int(y1): int(y2), int(x1): int(x2), :])
            os.remove(img_file)
            os.remove(txt)


if __name__ == "__main__":
    orig_root_dir = r'/media/chen/wt/data/xueba_answer/orig_data/batch3'
    dst_root_dir = r'/media/chen/wt/data/xueba_answer/xueba_answer_data/batch3'
    mk_data_from_orig_pic(orig_root_dir, dst_root_dir)
    # unzip_cleaned_files(r'/media/chen/wt/data/hw_train_data_clean/batch_2_05')
    generate_label_file(r'/media/chen/wt/data/hw_train_data_clean/batch_3')
    # post_process(r'/media/chen/wt/data/hw_train_data_clean/batch_2')
    # postprocess_cleaned_files(r'/media/chen/wt/data/hw_train_data_clean/batch_2')
    # tar_hw_general_hard_data()
    # tar_label_data()
