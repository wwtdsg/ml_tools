from baidubce.services.bos.bos_client import BosClient
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
import logging
import os
import tqdm
import tempfile
import hashlib
import tarfile
import base64
from concurrent.futures import ThreadPoolExecutor
import time
import io


class Config:
    def __init__(self, object_dir):
        self.bucket_name = 'ayx-public'
        self.object_dir = object_dir
        self.access_key_id = '05da87abbd65454c86d3ab477b145238'
        self.secret_access_key = 'c10d2970728e402289403792400c77ce'


def bos_config(access_key_id, secret_access_key, bos_host="bj.bcebos.com"):
    # 设置日志文件的句柄和日志级别
    logger = logging.getLogger('baidubce.http.bce_http_client')
    fh = logging.FileHandler("bos_download.log")
    fh.setLevel(logging.DEBUG)

    # 设置日志文件输出的顺序、结构和内容
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    # 创建BceClientConfiguration
    config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
    return config


def get_file_md5_string(file):
    md5_string = get_file_md5_object(file).hexdigest().lower()
    return md5_string


def get_file_md5_object(file):
    if not os.path.exists(file):
        return ''
    md5_object = hashlib.md5()
    with open(file, 'rb') as f:
        md5_object.update(f.read())

    return md5_object


def get_data_md5_object(data):
    md5_object = hashlib.md5()
    md5_object.update(data)
    return md5_object


def download_model_file(config, file_name, save_path, md5_string):
    """
    :param config: 基本配置内容，需要包括：
        bucket_name：命名空间
        object_dir：bos中的对象路径
        access_key_id：
        secret_access_key：
    :param file_name: 对象名
    :param save_path: 对象本地存储路径
    :param md5_string: 对象的md5校验码
    :return:
    """
    bucket_name = config.bucket_name
    object_dir = config.object_dir
    access_key_id = config.access_key_id
    secret_access_key = config.secret_access_key

    object_key = os.path.join(object_dir, file_name)
    save_file_path = os.path.join(save_path, file_name)

    client_config = bos_config(access_key_id, secret_access_key)
    # 新建BosClient
    bos_client = BosClient(client_config)

    if os.path.exists(save_file_path):
        os.unlink(save_file_path)
    bos_client.get_object_to_file(bucket_name, object_key, save_file_path)  # 直接下载完整模型文件

    if get_file_md5_string(save_file_path) == md5_string.lower():
        return True
    else:
        print('Download whole file failed, trying to download it tile by tile')
        tile_size = 1024 * 1024

        try:  # 获取文件大小
            content_length = int(bos_client.get_object_meta_data(bucket_name, object_key).metadata.content_length)
        except ValueError:
            print("BOS content length Error")
            return False

        with open(save_file_path, 'wb') as f:
            for i in tqdm.tqdm(range(0, content_length, tile_size)):
                tmp_f = tempfile.NamedTemporaryFile()
                bos_client.get_object_to_file(bucket_name, object_key, tmp_f.name, (i, i + tile_size - 1))
                f.write(tmp_f.read())
                tmp_f.close()

        if get_file_md5_string(save_file_path) != md5_string.lower():
            print('MD5 mismatch, download failed..')
            return False
        return True


def upload_file_to_bos(config, file):
    """
    直接上传文件到BOS，仅支持几十兆以内的小文件，如果太大请使用append上传
    :param config:
    :param file:
    :return:
    """
    bucket_name = config.bucket_name
    object_dir = config.object_dir
    access_key_id = config.access_key_id
    secret_access_key = config.secret_access_key

    object_key = os.path.join(object_dir, file)

    client_config = bos_config(access_key_id, secret_access_key)
    # 新建BosClient
    bos_client = BosClient(client_config)

    content_length = os.path.getsize(file)
    md5_object = get_file_md5_object(file)

    tar_file = open(file, 'rb')
    data = tar_file.read()
    content_md5 = base64.b64encode(md5_object.digest())
    res = bos_client.put_object(bucket_name, object_key, data, content_length, content_md5)
    if res.metadata.content_md5 != str(content_md5, encoding="utf-8"):
        print('{} upload_failed'.format(file))
        return False
    else:
        os.unlink(file)
        return True


def append_data_to_bos(bos_client, config, dst_file, src_data, src_data_size, next_append_offset=0):
    """
    使用append方式分片上传，仅支持5G以内的文件，如果超过5G请自行分割打包后上传
    :param bos_client:
    :param config:
    :param dst_file:
    :param src_data:
    :param src_data_size:
    :param next_append_offset:
    :return:
    """
    bucket_name = config.bucket_name
    object_dir = config.object_dir
    object_key = os.path.join(object_dir, dst_file)

    content_length = src_data_size
    md5_object = get_data_md5_object(src_data)
    content_md5 = base64.b64encode(md5_object.digest())
    if next_append_offset:
        res = bos_client.append_object(bucket_name, object_key, src_data, content_md5, content_length,
                                       offset=next_append_offset)
    else:
        res = bos_client.append_object(bucket_name, object_key, src_data, content_md5, content_length)
    return res.metadata.content_md5, int(res.metadata.bce_next_append_offset)


def tar_training_data_and_upload(config, training_data, dst_tar_file, dst_training_file):
    tar = tarfile.open(dst_tar_file, "w:tar")
    new_training_data = []

    for line in training_data:
        img_path, gt = line.split(' ', 1)
        new_img_file = '_'.join(img_path.split('/')[-3:])
        tar.add(img_path, new_img_file, recursive=False)
        new_training_data.append(new_img_file + ' ' + gt)

    with open(dst_training_file, 'w') as f:
        f.writelines(new_training_data)
    tar.add(dst_training_file)
    tar.close()

    if upload_file_to_bos(config, dst_tar_file):
        os.unlink(dst_training_file)


def tar_dir_and_upload(config, src_dir, arc_name, dst_tar_file):
    """
    打包目录并上传，仅支持不超过5G的文件
    :param config:
    :param src_dir:
    :param arc_name:
    :param dst_tar_file:
    :return:
    """
    tile_size = 1024 * 1024 * 10
    with tarfile.open(dst_tar_file, "w:gz") as tar:
        print("tar file...")
        tar.add(src_dir, arcname=arc_name)
    file_size = os.path.getsize(dst_tar_file)
    if file_size > 1024 * 1024 * 1024 * 5:
        print("Error: tar file too large")
        return

    access_key_id = config.access_key_id
    secret_access_key = config.secret_access_key

    client_config = bos_config(access_key_id, secret_access_key)
    # 新建BosClient
    bos_client = BosClient(client_config)
    offset = 0
    md5_str = ''
    with open(dst_tar_file, 'rb') as f:
        for _ in tqdm.tqdm(range(0, file_size, tile_size)):
            data = f.read(tile_size)
            md5_str, offset = append_data_to_bos(bos_client, config, dst_tar_file, data, len(data), offset)
        assert f.tell() == file_size
    md5_object = get_file_md5_object(dst_tar_file)
    content_md5 = base64.b64encode(md5_object.digest())
    if md5_str != str(content_md5, encoding="utf-8"):
        print("MD5 mismatch, upload failed")
    else:
        os.unlink(dst_tar_file)


def tar_from_training_file_and_upload(config, training_file, dataset_name, tile_count=10000):
    with open(training_file) as f:
        training_data = f.readlines()

    pool = ThreadPoolExecutor(8)
    tasks = []

    for i in range(7770000, len(training_data), tile_count):
        dst_tar_file = dataset_name + '_%03d' % (i // tile_count) + r'.tar'
        dst_training_file = dataset_name + '_%03d' % (i // tile_count) + r'.txt'
        tile_training_data = training_data[i: i + tile_count]
        # tar_and_upload(config, tile_training_data, dst_tar_file, dst_training_file)
        tasks.append(
            pool.submit(tar_training_data_and_upload, config, tile_training_data, dst_tar_file, dst_training_file))
    task_done = 0
    while task_done < len(tasks):
        task_status = [task.done() for task in tasks]
        task_done = sum(task_status)
        print('done/total: %d/%d, task left: %d' % (task_done, len(tasks), len(tasks) - task_done))
        time.sleep(5)
    return


def upload_hw_name_dataset():
    config = Config('ai/dataset/hw_name')
    training_file = r'/media/chen/wt/handwritekaohao/filter_image/rel_gen_blankname_0826_ssd.txt'
    dataset_name = r'hw_name'
    tar_from_training_file_and_upload(config, training_file, dataset_name)


def upload_formula_det_dataset():
    config = Config('ai/dataset/formula/detection')

    val_dir = r'/media/chen/hcn/data/pixel_link_data/200413/val'
    tar_dir_and_upload(config, val_dir, 'val', 'val.tar')

    train_dir = r'/media/chen/hcn/data/pixel_link_data/200413/train_quad'
    tar_dir_and_upload(config, train_dir, 'train_quad', 'train_quad.tar')


def upload_ocr_recognition_dataset():
    config = Config('ai/dataset/ocr/recognition')
    training_file = r'/media/chen/wt/handwritekaohao/filter_image/rel_gen_blankname_0826_ssd.txt'
    dataset_name = r'ocr_recognition_train'
    tar_from_training_file_and_upload(config, training_file, dataset_name)


if __name__ == '__main__':
    dataset = 'formula_det'
    if dataset == 'hw_name':
        upload_hw_name_dataset()
    elif dataset == 'formula_det':
        upload_formula_det_dataset()
    elif dataset == 'ocr_recg':
        pass
    # download_model_file(config, 'epoch_4_2000_resctcmodel.pth', './', 'd9c00e86e57166267f7c10a170036a07')
