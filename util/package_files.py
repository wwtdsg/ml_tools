import os
import glob
import shutil
import zipfile
import random
import sys


def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()


def make_zip_from_listfile(listfile, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    test_count = 0
    with open(listfile) as f:
        for line in f.readlines():
            file = line.strip()
            arcname = os.path.split(file)
            zipf.write(file, arcname)
            test_count += 1
            if test_count > 10:
                break
    zipf.close()


def pack_files(img_paths, dst_root_path, pack_num):
    # img_paths = glob.glob(base_path + "/*.png")
    for index, img_path in enumerate(img_paths):
        dst_path = os.path.join(dst_root_path, str(index // pack_num))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        root_path, img_file = os.path.split(img_path)
        subject = root_path.split('/')[-3]
        sub_id = root_path.split('/')[-1]
        img_file = "{}_{}_{}".format(subject, sub_id, img_file)
        # txt_path = img_path.replace(".png", ".txt")
        # txt_file = img_file.replace(".png", ".txt")
        shutil.copyfile(img_path, os.path.join(dst_path, img_file))
        # shutil.copyfile(txt_path, os.path.join(dst_path, txt_file))
        if (index + 1) % pack_num == 0:
            make_zip(dst_path, os.path.join(dst_root_path, dst_path + ".zip"))


def test_validate(dst_dir):
    if not os.path.exists(dst_dir):
        return None
    if os.path.exists(os.path.join(dst_dir, "info.txt")):
        os.remove(os.path.join(dst_dir, "info.txt"))
    txt_files = glob.glob(dst_dir + '/*.txt')
    err_null_count = 0
    err_val_count = 0
    total_txt_count = 0
    for txt in txt_files:
        with open(txt) as f:
            lines = f.readlines()
            if not lines:
                os.remove(txt)
                continue
            total_txt_count += 1
            for line in lines:
                try:
                    ty = int(line.split(",")[4])
                except ValueError:
                    print('NullError', '/'.join(txt.split('/')[-2:]))
                    err_null_count += 1
                    break
                if ty != 1 and ty != 2:
                    print('LabelError', '/'.join(txt.split('/')[-2:]), ty)
                    err_val_count += 1
    print("total error labels:", err_val_count + err_null_count)
    print("err_null_count: %d, err_val_count: %d" % (err_null_count, err_val_count))
    print("total_txt_count:", total_txt_count)
    return err_val_count + err_null_count


def pack_pic_form_files():
    subjects = ['biology', 'chinese', 'geology', 'math', 'physic', 'chemical', 'english', 'history', 'moral']
    root_dir = r'/media/chen/chenqq/datas'
    img_paths = []
    for subject in subjects:
        id_dir = os.path.join(root_dir, subject, "origin_papers")
        # for id_name in os.listdir(id_dir):
        [img_paths.extend(glob.glob(os.path.join(id_dir, id_name) + "/*.png")) for id_name in os.listdir(id_dir)]
    random.shuffle(img_paths)
    pack_files(img_paths, '/media/chen2/data/wt/pic_form_data', 300)


def crop_train_data(root_path, size=512):
    txt_path = []
    for dir in os.listdir(root_path):
        txt_path.extend(glob.glob(os.path.join(root_path, dir) + "/*.txt"))
    img_path = [txt.replace("txt", "png") for txt in txt_path]

    pass


if __name__ == '__main__':
    make_zip_from_listfile("/home/chen/lpj/dataset/ocr_print/labels/ocr_data_tobe_move.txt",
                           "/home/chen/lpj/dataset/ocr_print/labels/ocr_data_zip.zip")
    pass
