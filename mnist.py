import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models import mobilenet_v2, resnet34
import numpy as np
import cv2
from scipy.stats import entropy
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import PIL
import random
import tqdm
import glob
import time
import math
from collections import Counter

img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


def get_train_transform(mean=img_mean, std=img_std, size=0):
    train_transform = transforms.Compose([
        # RandomGaussianBlur(),
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform


class SimpleClassDataset(Dataset):
    def __init__(self, label_file, class_file, input_size=64, is_train=True):
        self.img_list = []
        self.label_list = []
        with open(class_file) as f:
            lines = f.readlines()
            self.class_dict = {line.strip(): idx for idx, line in enumerate(lines)}
        with open(label_file) as f:
            lines = f.readlines()
        for line in lines:
            img, label = line.strip().split(' ', 1)
            self.img_list.append(img)
            self.label_list.append(self.class_dict[label])
        self.transform = get_train_transform(size=input_size)
        self.use_gpu = torch.cuda.is_available()
        self.is_train = is_train

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        cls = self.label_list[index]
        # img = cv2.imread(img_path)
        pil_img = Image.open(img_path).convert('RGB')
        img = self.transform(pil_img)
        # img_np = torch.from_numpy(img).permute(2, 0, 1)
        # cls_np = torch.from_numpy(cls)
        if self.use_gpu:
            return img.cuda(), torch.from_numpy(np.array(int(cls))).cuda()
        else:
            return img, torch.from_numpy(np.array(int(cls)))

    # @staticmethod
    # def collate_fn(batch):
    #     imgs = []
    #     classes = []
    #     for img, cls in batch:
    #         imgs.append(img[np.newaxis, :, :, :])
    #         classes.append(cls[np.newaxis, :, :])
    #     imgs = np.concatenate((imgs), axis=0)
    #     classes = np.concatenate((classes), axis=0)
    #     return imgs, classes


# Training settings


def creat_dataset(train_root, test_root, class_file, batch_size):
    # MNIST Dataset
    train_dataset = SimpleClassDataset(train_root, class_file)

    test_dataset = SimpleClassDataset(test_root, class_file)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, )
    # collate_fn=train_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self, num_classes=136):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        # self.backbone = mobilenet_v2()
        self.backbone = resnet34()
        # fc_features = self.backbone.classifier[1].in_features
        fc_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_features, num_classes)
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        # self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc = nn.Linear(1440, 136)

    def forward(self, x):
        out = self.backbone(x)
        # in_size = 64
        # in_size = x.size(0)  # one batch
        # # x: 64*10*12*12
        # x = F.relu(self.mp(self.conv1(x)))
        # # x: 64*20*4*4
        # x = F.relu(self.mp(self.conv2(x)))
        # x = F.relu(self.mp(self.conv3(x)))
        # # x: 64*320
        # x = x.view(in_size, -1)  # flatten the tensor
        # # x: 64*10
        # x = self.fc(x)
        return out


def train(model, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        log_softmax = F.log_softmax(output, dim=1)
        loss = F.nll_loss(log_softmax, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mnist_test():
    batch_size = 64
    train_file = r'/media/chen/wt/tmp/kk/data/new_train_label.txt'
    val_file = r'/media/chen/wt/tmp/kk/data/new_val_label.txt'
    class_file = r'/media/chen/wt/tmp/kk/data/new_classes.txt'
    train_loader, test_loader = creat_dataset(train_file, val_file, class_file, batch_size)
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    resume_epoch = 10
    if resume_epoch:
        stat_dict = torch.load('epoch_%d.pth' % resume_epoch)
        model.load_state_dict(stat_dict)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, 50):
        train(model, train_loader, optimizer, epoch)
        with torch.no_grad():
            test(model, test_loader)
            if epoch % 5 == 0:
                save_filename = 'epoch_%d.pth' % epoch
                torch.save(model.state_dict(), save_filename)


def infer(img_root):
    class_file = r'/media/chen/wt/tmp/kk/data/new_classes.txt'
    with open(class_file) as f:
        lines = f.readlines()
        num_class = {idx: line.strip() for idx, line in enumerate(lines)}
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    transform = get_train_transform(size=64)
    resume_epoch = 10
    stat_dict = torch.load('epoch_%d.pth' % resume_epoch)
    model.load_state_dict(stat_dict)
    model.eval()
    font = ImageFont.truetype('./fonts/Deng.ttf', 20)
    # test_img_file = dsg_img
    for file in os.listdir(img_root):
        # detect_boxes = kgt_area_detect(test_img_file)
        if file.endswith('txt'):
            continue
        test_img_file = os.path.join(img_root, file)
        detect_boxes = kh_area_detect(test_img_file)

        img_prefix = os.path.splitext(os.path.split(test_img_file)[-1])[0]
        txt_file = os.path.join(img_root, img_prefix + '.txt')
        f = open(txt_file, 'w')
        lines = []

        with torch.no_grad():
            # with open(val_file) as f:
            #     lines = f.readlines()
            orig_img = Image.open(test_img_file).convert('RGB')
            orig_w, orig_h = orig_img.size
            vis_img = Image.new(orig_img.mode, (orig_w * 2, orig_h), (255, 255, 255))
            vis_img.paste(orig_img, (0, 0))
            txt_draw = ImageDraw.Draw(vis_img)
            for rect in detect_boxes:
                # img_file, gt = line.strip().split(' ', 1)
                # img = np.array(Image.open(img_file).convert('RGB'))
                # img = torch.from_numpy(img)
                crop_img = orig_img.crop(rect)
                w, h = crop_img.size
                padding_size = 10
                padding_crop_img = Image.new(crop_img.mode, (w + padding_size, h + padding_size), (255, 255, 255))
                padding_crop_img.paste(crop_img, (5, 5))
                img = transform(padding_crop_img)
                # img = torch.from_numpy(np.asarray(padding_crop_img, dtype=np.float32)).permute(2, 0, 1)
                img = img[np.newaxis, :].cuda()
                output = model(img)
                # test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred_probs = F.softmax(output, dim=1).cpu().numpy().squeeze()
                # pred = output.data.max(1, keepdim=True)[1]
                pred = np.argmax(pred_probs)
                pred_prob = pred_probs[pred]
                # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                print(num_class[pred], pred_prob)
                if pred_prob < 0.8 or num_class[pred] == '14':
                    continue
                # padding_crop_img.save('crop_img.png')
                txt_draw.text((rect[0] + orig_w, rect[1]), num_class[pred], fill=6, font=font)
                txt_draw.rectangle((rect[0], rect[1], rect[2], rect[3]), outline=(0, 255, 0))
                lines.append('%d, %d, %d, %d, %s\n' % (rect[0], rect[1], rect[2], rect[3], num_class[pred]))
            vis_img.save('/media/chen/wt/tmp/kk/rel.png')
        f.writelines(lines)
        f.close()


def rotate_bound(image, rotate_angle, pad_value=(0, 0, 0)):
    """
    旋转图片，超出原图大小的边界用指定颜色填充.
    :param image: 待旋转图片
    :param rotate_angle:  旋转角度
    :param pad_value: 边界填充颜色
    :return:
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -rotate_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=pad_value)


def correct_img(src_image):
    """
    倾斜校正.
    :param src_image: 待校正图片
    :return:
    """
    if src_image.ndim == 3:
        gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = src_image

    _, bin_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    thetas = np.linspace(-2, 2., 50, endpoint=False)
    entropys = []
    for angle in thetas:
        rotated_image = rotate_bound(bin_image, angle)

        sum_row = np.sum(rotated_image, axis=1)
        if not float(np.sum(sum_row)):
            continue
        sum_row = sum_row / float(np.sum(sum_row))
        entropys.append(entropy(sum_row))

    if not entropys:
        return src_image, 0
    else:
        angle = thetas[np.argmin(entropys)]

    # (-00, -0.2)U(0.2, +00)范围内旋转, [0, 0.2]不旋转
    if abs(angle) > 0.2:
        ret_image = rotate_bound(src_image, angle, (255, 255, 255))
        return ret_image, angle

    return src_image, angle


def kh_area_detect(img_file):
    root_dir = os.path.split(img_file)[0]
    src_img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img, angle = correct_img(gray_img)
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(root_dir, 'thresh_img.png'), thresh_img)
    img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, np.ones((1, 11), np.uint8))
    cv2.imwrite(os.path.join(root_dir, 'open.png'), img)
    _, contours, _ = cv2.findContours(255 - img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_area = gray_img.shape[0] * gray_img.shape[1]

    rel_cnts = []
    rel_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x - 1, y - 1, w + 2, h + 2
        cnt_area = w * h
        if cnt_area > img_area // 50 or cnt_area < img_area // 8000 \
                or w < 5 or w > 100 or h < 10 or h > 100 or w / h > 3 or h / w > 3:
            continue
        rel_areas.append(cnt_area)
        rel_cnts.append([x, y, x + w, y + h])
        # cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imwrite('/media/chen/wt/tmp/kk/boxes.png', src_img)
    return rel_cnts


def kgt_area_detect(img_file):
    root_dir = os.path.split(img_file)[0]
    src_img, angle = correct_img(cv2.imread(img_file))
    cv2.imwrite(os.path.join(root_dir, 'correct_img.png'), src_img)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((1, 2), np.uint8))
    cv2.imwrite(os.path.join(root_dir, 'MORPH_CLOSE.png'), img)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 8), np.uint8))
    cv2.imwrite(os.path.join(root_dir, 'MORPH_OPEN.png'), img)
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(root_dir, 'threshold.png'), thresh_img)
    _, contours, _ = cv2.findContours(255 - thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_area = img.shape[0] * img.shape[1]

    rel_cnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x - 2, y - 2, w + 1, h + 1
        cnt_area = w * h
        if cnt_area > img_area // 50 or cnt_area < img_area // 3500:  # 后一个阈值可以取900-2000
            continue
        rel_cnts.append([x, y, x + w, y + h])
        cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite('/media/chen/wt/tmp/kk/boxes.png', src_img)
    return rel_cnts


def random_noise(height, width):
    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add random noise
    for _ in range(random.randint(0, 1000)):
        image[random.randint(0, height - 1), random.randint(0, width - 1)] = random.randint(50, 254)
    return Image.fromarray(image).convert('L')


def generate_data(save_path, data_nums):
    tihao_nums = [str(i) for i in range(1, 100)]
    kaohao_nums = ['[%d]' % i for i in range(10)]
    xuanxiang_cs = ['[%s]' % chr(i) for i in range(ord('A'), ord('Z') + 1)]
    fonts = [os.path.join('./fonts', font) for font in os.listdir('./fonts')]
    texts = tihao_nums + kaohao_nums + xuanxiang_cs
    with open(os.path.join(save_path, 'classes.txt'), 'w') as f:
        classes = [item + '\n' for item in texts if item != '[O]']
        f.writelines(classes)
    with open(os.path.join(save_path, 'label.txt'), 'w') as f:
        label_lines = []
        for i in tqdm.tqdm(range(data_nums)):
            # image_font = ImageFont.truetype(font=random.choice(fonts), size=32)
            for j, text in enumerate(texts):
                image_font = ImageFont.truetype(font=random.choice(fonts), size=random.randint(20, 32))
                text_width, text_height = image_font.getsize(text)
                padding_left, padding_right = random.randint(1, 10), random.randint(1, 10)
                padding_top, padding_bottom = random.randint(1, 10), random.randint(1, 10)
                txt_img = Image.new('L',
                                    (text_width + padding_left + padding_right,
                                     text_height + padding_top + padding_bottom),
                                    255)
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((padding_left, padding_top // 2), text, fill=random.randint(1, 40), font=image_font)
                random_angle = (random.randint(int(0 - 0.2 * 100), int(0.2 * 100))) / 100
                rotated_img = txt_img.rotate(random_angle, expand=1)
                if random.randint(1, 10) < 4:
                    final_image = rotated_img.filter(ImageFilter.GaussianBlur(radius=True))
                else:
                    final_image = rotated_img
                img_name = save_path + '/%d_%d.jpg' % (i, j)
                final_image.save(img_name)
                label_lines.append(img_name + ' ' + text + '\n')
        f.writelines(label_lines)
    random.shuffle(label_lines)
    val_lines = label_lines[:len(label_lines) // 10]
    train_lines = label_lines[len(label_lines) // 10:]
    with open(os.path.join(save_path, 'train_label.txt'), 'w') as f:
        f.writelines(train_lines)
    with open(os.path.join(save_path, 'val_label.txt'), 'w') as f:
        f.writelines(val_lines)
    pass


def generate_reality_data(img_file, dst_root, prefix):
    detect_boxes = kh_area_detect(kh_img_file_path)
    orig_img = Image.open(kh_img_file_path).convert('RGB')
    kh_labels = []
    for i, rect in enumerate(detect_boxes):
        crop_img = orig_img.crop(rect)
        w, h = crop_img.size
        padding_size = 10
        padding_crop_img = Image.new(crop_img.mode, (w + padding_size, h + padding_size), (255, 255, 255))
        padding_crop_img.paste(crop_img, (5, 5))
        save_img_path = os.path.join(dst_root, '%s_%d_%d.png' % (prefix, 9 - i // 7, i))
        label = save_img_path + ' ' + '[%d]' % (9 - i // 7) + '\n'
        kh_labels.append(label)
        padding_crop_img.save(save_img_path)
    with open(os.path.join(dst_root, '%s_label.txt' % prefix), 'w') as f:
        f.writelines(kh_labels)


def split_reality_data(img_file, label_file, dst_root, prefix):
    orig_img = Image.open(img_file).convert('RGB')
    kgt_labels = []
    with open(label_file) as f:
        for i, line in enumerate(f.readlines()):
            items = line.strip().split(',')
            rect = list(map(int, items[:4]))
            gt = items[-1].strip()

            crop_img = orig_img.crop(rect)
            w, h = crop_img.size
            padding_size = 10
            padding_crop_img = Image.new(crop_img.mode, (w + padding_size, h + padding_size), (255, 255, 255))
            padding_crop_img.paste(crop_img, (5, 5))
            save_img_path = os.path.join(dst_root, '%s_%d.png' % (prefix, i))
            label = save_img_path + ' ' + gt + '\n'
            kgt_labels.append(label)
            padding_crop_img.save(save_img_path)
    with open(os.path.join(dst_root, '%s_label.txt' % prefix), 'w') as f:
        f.writelines(kgt_labels)
    random.shuffle(kgt_labels)
    label_count = len(kgt_labels)
    val_kgt_labels = kgt_labels[:label_count // 10]
    train_kgt_labels = kgt_labels[label_count // 10:]
    with open(os.path.join(dst_root, 'new_train_label.txt'), 'a') as f:
        f.writelines(train_kgt_labels)
    with open(os.path.join(dst_root, 'new_val_label.txt'), 'a') as f:
        f.writelines(val_kgt_labels)


def generate_yolov3_labels():
    pass


def label_img_2_yolov3():
    pass


def crop_orig_paper(sub_id, kh_rect, kgt_rect, dst_dir):
    paper_id_root = '/media/chen/wt/download_paper/paper'
    img_path = os.path.join(paper_id_root, sub_id, 'src')
    imgs = glob.glob(img_path + '/*.png')
    x1_kh, y1_kh, x2_kh, y2_kh = kh_rect
    x1_kg, y1_kg, x2_kg, y2_kg = kgt_rect if kgt_rect else (None, None, None, None)
    for i, img_path in enumerate(random.sample(imgs, 10)):
        img = cv2.imread(img_path)
        img_name = os.path.split(img_path)[-1]

        dst_save_kh = os.path.join(dst_dir, sub_id + "_kh_%d" % i + img_name)
        kh_img = img[y1_kh:y2_kh, x1_kh: x2_kh]
        cv2.imwrite(dst_save_kh, kh_img)
        if kgt_rect:
            dst_save_kg = os.path.join(dst_dir, sub_id + "_kg_%d" % i + img_name)
            kg_img = img[y1_kg:y2_kg, x1_kg: x2_kg]
            cv2.imwrite(dst_save_kg, kg_img)


def label_with_template(root_dir, template_txt):
    prefix_id = ''.join(template_txt.split('_')[:2])
    relative_txts = [file for file in os.listdir(root_dir) if
                     file.endswith('txt') and ''.join(file.split('_')[:2]) == prefix_id]
    with open(os.path.join(root_dir, template_txt)) as f:
        lines = f.readlines()
    for txt in relative_txts:
        with open(os.path.join(root_dir, txt), 'w') as f:
            f.writelines(lines)


def morphoy_process(binary_img, kernel_close, kernel_open):
    binary_img = binary_img.astype(np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    return opening


def locate_ctl_point(img_file):
    img_root_dir = os.path.split(img_file)[0]
    img = cv2.imread(img_file)
    img, _, _ = cal_roi(img, img_root_dir)
    cv2.imwrite(os.path.join(img_root_dir, 'roi_vis.png'), img)
    r_img = img[:, :, -1]
    b_mask = img[:, :, 0] < 90
    g_mask = img[:, :, 1] < 90
    r_mask = img[:, :, 2] > 100
    red_mask = r_mask & b_mask & g_mask
    cv2.imwrite(os.path.join(img_root_dir, 'red_mask.png'), red_mask * 255)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = (gray_img < 50) * 255
    cv2.imwrite(os.path.join(img_root_dir, 'black_mask.png'), black_mask)
    kernel_close = np.ones((3, 3), np.uint8)  # 给图像闭运算定义核
    kernel_open = np.ones((3, 3), np.uint8)  # 给图像开运算定义核

    morphy_black = morphoy_process(black_mask, kernel_close, kernel_open)
    cv2.imwrite(os.path.join(img_root_dir, 'morphy_black.png'), morphy_black)

    morphy_red = morphoy_process(red_mask, kernel_close, kernel_open)
    cv2.imwrite(os.path.join(img_root_dir, 'morphy_red.png'), morphy_red)


def cal_roi(img, img_root_dir):
    h, w = img.shape[:-1]
    scale_rate = 6
    dst_h, dst_w = h // scale_rate, w // scale_rate
    img = cv2.bilateralFilter(img, 11, 50, 50)
    cv2.imwrite(os.path.join(img_root_dir, 'shuangbian.png'), img)
    r_g_mask = (img[:, :, 2].astype(np.int) - img[:, :, 1].astype(np.int)) > 50
    r_b_mask = (img[:, :, 2].astype(np.int) - img[:, :, 0].astype(np.int)) > 40
    r_mask = img[:, :, 2] > 120
    r_img = ((r_g_mask & r_b_mask & r_mask) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(img_root_dir, 'r_img.png'), r_img)

    # r_img = img[:, :, 2]
    # _, r_img = cv2.threshold(r_img, 128, 255, cv2.THRESH_BINARY)
    # img[:, :, 2] = r_img
    # cv2.imwrite(os.path.join(img_root_dir, 'red_plus.png'), img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(img_root_dir, 'gray_img.png'), gray_img)
    ada_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    cv2.imwrite(os.path.join(img_root_dir, 'ada_thresh.png'), ada_img)
    shrink_img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(img_root_dir, 'shrink_img.png'), shrink_img)
    gray_img = cv2.cvtColor(shrink_img, cv2.COLOR_BGR2GRAY)
    # black_mask = (gray_img < 30)
    # b_mask = shrink_img[:, :, 0] < 80
    # g_mask = shrink_img[:, :, 1] < 80
    r_g_mask = (shrink_img[:, :, 2].astype(np.int) - shrink_img[:, :, 1].astype(np.int)) > 50
    r_b_mask = (shrink_img[:, :, 2].astype(np.int) - shrink_img[:, :, 0].astype(np.int)) > 40
    r_mask = shrink_img[:, :, 2] > 120
    # red_mask = r_g_mask & r_b_mask
    black_coord = np.argwhere(gray_img < 50)
    red_coord = np.argwhere(r_g_mask & r_b_mask & r_mask)
    min_dist = float('inf')
    pix_pair = None
    near_length = 2
    for red_pix in red_coord:
        y_min, y_max = max(red_pix[0] - near_length, 0), min(red_pix[0] + near_length, dst_h - 1)
        x_min, x_max = max(red_pix[1] - near_length, 0), min(red_pix[1] + near_length, dst_w - 1)
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if gray_img[y, x] < 50 and x != red_pix[1] and y != red_pix[0]:
                    square_l = np.square(red_pix[0] - y) + np.square(red_pix[1] - x)
                    if min_dist > square_l:
                        min_dist = square_l
                        pix_pair = np.array([np.array([y, x]) * scale_rate, red_pix * scale_rate], dtype=np.int)
                        center_pix = (np.array([y, x]) * scale_rate + red_pix * scale_rate) // 2
        # for coord in zip(range(y_min, y_max), range(x_min, x_max)):
        #     x_near, y_near = np.abs(red_pix - black_pix) < 3
        #     if not x_near or not y_near:
        #         continue
        #     dist = np.linalg.norm(red_pix - black_pix)
        #     if dist < min_dist:
        #         min_dist = dist
        #         pix_pair = np.array([black_pix * scale_rate, red_pix * scale_rate], dtype=np.int)
        #         center_pix = (black_pix * scale_rate + red_pix * scale_rate) // 2
    print(min_dist, pix_pair)
    box_width = int(min_dist * scale_rate / 2)
    # y1, x1, y2, x2 = np.min(pix_pair[:, 0]), np.min(pix_pair[:, 1]), np.max(pix_pair[:, 0]), np.max(pix_pair[:, 1])
    y1, x1 = center_pix[0] - box_width, center_pix[1] - box_width
    y2, x2 = center_pix[0] + box_width, center_pix[1] + box_width
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.rectangle(img, (x1 - box_width, y1 - box_width), (x2 + box_width, y2 + box_width), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(img_root_dir, 'roi_vis.png'), img)
    return img, (x1 - box_width, y1 - box_width), (x2 + box_width, y2 + box_width)


def parent_contour_chk(cnt):
    area = cv2.contourArea(cnt)
    if area > 400 or area < 50:
        return False
    perimeter = cv2.arcLength(cnt, True)
    rect = cv2.minAreaRect(cnt)

    rect_perimeter = 2 * (sum(rect[1]))
    hull_perimeter = cv2.arcLength(cv2.convexHull(cnt), True)

    if abs(rect[1][0] - rect[1][1]) > 6 \
            or abs(rect_perimeter - perimeter) > 5 \
            or abs(perimeter - hull_perimeter) > 6:
        return False
    return True


def child_contour_chk(cnt, gray_img):
    # M = cv2.moments(cnt)
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    # print(cx, cy)
    mask = np.zeros(gray_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    roi = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    pixel_sum = np.sum(roi)
    pixel_count = cv2.countNonZero(roi)
    ave_pixel_val = pixel_sum / pixel_count

    # rect = cv2.boundingRect(cnt)
    #
    # rect_perimeter = 2 * (sum(rect[1]))
    # hull_perimeter = cv2.arcLength(cv2.convexHull(cnt), True)
    #
    # if abs(rect[1][0] - rect[1][1]) > 2 \
    #         or abs(rect_perimeter - perimeter) > 5 \
    #         or abs(perimeter - hull_perimeter) > 3:
    #     return False
    if ave_pixel_val > 210:
        return True
    else:
        return False


def get_roi_img(cnt, gray_img):
    # new_cnt = np.zeros_like(cnt)
    rect = cv2.boundingRect(cnt)
    x1, y1 = rect[0], rect[1]
    x2, y2 = rect[0] + rect[2], rect[1] + rect[3]
    roi_img = gray_img[y1: y2, x1: x2]
    # new_cnt[:, 0, 0] = cnt[:, 0, 0] - x1 + 1
    # new_cnt[:, 0, 1] = cnt[:, 0, 1] - y1 + 1
    return roi_img  # , new_cnt, (x1, y1)


def is_corner(gray_img, parent_cnt, child_cnt):
    # if rect[2] <

    mask_out = np.zeros(gray_img.shape[:2], np.uint8)
    cv2.drawContours(mask_out, [parent_cnt], -1, 255, -1)
    mask_in = np.ones(gray_img.shape[:2], np.uint8) * 255
    cv2.drawContours(mask_in, [child_cnt], -1, 0, -1)

    mask_ring = cv2.bitwise_and(mask_out, mask_in)
    ring_pixel_count = cv2.countNonZero(mask_ring)

    mask_in = 255 - mask_in
    roi_in = cv2.bitwise_and(gray_img, gray_img, mask=mask_in)
    cv2.imwrite('roi_in.jpg', roi_in)

    pixel_sum = np.sum(roi_in)
    if pixel_sum is None:
        print('???')
    pixel_count = cv2.countNonZero(roi_in) + 0.0001
    ave_pixel_val = pixel_sum / pixel_count

    if ave_pixel_val < 180 or not (pixel_count * 3 < ring_pixel_count < pixel_count * 6):
        return False
    return True


def contour_chk(contour, thresh_img):
    if not parent_contour_chk(contour):
        return False
    # gray_img, parent_cnt, (x1, y1) = get_roi_img(contour, gray_img)
    thresh_img = get_roi_img(contour, thresh_img)
    if debug:
        cv2.imwrite('roi_img.png', thresh_img)
    # _, thresh_img = cv2.threshold(thresh_img, 100, 255, cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)
    for i, cnt in enumerate(cnts):
        if hierarchy[i][2] == -1 and hierarchy[i][3] != -1:
            parent_idx = hierarchy[i][3]
            parent_cnt = cnts[parent_idx]
            parent_area = cv2.contourArea(parent_cnt)
            child_area = cv2.contourArea(cnt)
            area_rate = child_area / parent_area
            print('area rate', child_area / parent_area)
            # if is_corner(gray_img, parent_cnt, cnt):
            if 0.1 < area_rate < 0.3:
                return True
    return False


def check_red_flag(img, red_points):
    red_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(red_mask, [red_points.astype(np.int32)], -1, 255, 1)
    red_img = cv2.bitwise_and(img, img, mask=red_mask)
    red_pixel_count = cv2.countNonZero(cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY))
    r_value = np.sum(red_img[:, :, 2]) / red_pixel_count
    b_value = np.sum(red_img[:, :, 0]) / red_pixel_count
    g_value = np.sum(red_img[:, :, 1]) / red_pixel_count
    if debug:
        cv2.imwrite('red.png', red_img)

    if r_value - b_value > 30 and r_value - g_value > 30 and r_value > 100:
        print('yes, we find red flag')
        return True
    else:
        return False


def find_code_area(img, white_box, outer_box, code_box):
    edge1 = np.linalg.norm(outer_box[0, :] - outer_box[1, :])
    edge2 = np.linalg.norm(outer_box[1, :] - outer_box[2, :])
    if edge1 > edge2:
        red_points = np.round([white_box[0, :], white_box[3, :], outer_box[3, :], outer_box[0, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[1, :], outer_box[2, :], code_box[2, :], code_box[1, :]]).astype(np.int)
            return code_points, red_points
        red_points = np.round([white_box[1, :], white_box[2, :], outer_box[2, :], outer_box[1, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[0, :], outer_box[3, :], code_box[3, :], code_box[0, :]]).astype(np.int)
            return code_points, red_points
    else:
        red_points = np.round([white_box[2, :], white_box[3, :], outer_box[3, :], outer_box[2, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[0, :], outer_box[1, :], code_box[1, :], code_box[0, :]]).astype(np.int)
            return code_points, red_points
        red_points = np.round([white_box[0, :], white_box[1, :], outer_box[1, :], outer_box[0, :]]).astype(np.int)
        if check_red_flag(img, red_points):
            code_points = np.round([outer_box[2, :], outer_box[3, :], code_box[3, :], code_box[2, :]]).astype(np.int)
            return code_points, red_points
        # return code_points, red_points
    return None, red_points


def value_2_num(value):
    r_value = value[2]
    b_value = value[0]
    g_value = value[1]
    # 黑0，白1，红2，绿3
    if g_value - b_value > 50 and g_value - r_value > 50 and g_value > 100 and b_value < 140 and r_value < 140:
        return 3
    if r_value - b_value > 50 and r_value - g_value > 50 and r_value > 100 and b_value < 140 and g_value < 140:
        return 2
    if r_value > 140 and g_value > 140 and b_value > 140:
        return 1
    if r_value < 50 and g_value < 50 and b_value < 50:
        return 0
    ave_value = np.mean(value)
    if ave_value > 140:
        return 1
    return 0


def cvt_img_2_num(img):
    if debug:
        cv2.imwrite('code_num.jpg', img)
    arr_mean = np.mean(img, axis=0)
    num_count = [0] * 4
    # 黑0，白1，红2，绿3
    for value in arr_mean:
        num = value_2_num(value)
        num_count[num] += 1
    return np.argmax(num_count)


def crop_possible_region(ctl_box, img):
    ctl_point = ctl_box[0]
    w, h = ctl_box[1]
    ave_wh = np.mean(ctl_box[1])
    angle = ctl_box[2]
    gap = ave_wh / 8

    white_box = cv2.boxPoints((ctl_point, (w, h), angle))

    # 宽度增加
    outer_box = cv2.boxPoints((ctl_point, (w + gap * 2, h), angle))
    code_box = cv2.boxPoints((ctl_point, (w + gap * 9, h), angle))

    code_points, red_points = find_code_area(img, white_box, outer_box, code_box)
    if code_points is None:
        # 高度增加
        outer_box = cv2.boxPoints((ctl_point, (w, h + gap * 2), angle))
        code_box = cv2.boxPoints((ctl_point, (w, h + gap * 8), angle))
        code_points, red_points = find_code_area(img, white_box, outer_box, code_box)
        if code_points is None:
            print("Can't find red flag area")
            return img
    # if debug:
    #     test_img = np.copy(img)
    #     cv2.drawContours(test_img, [code_points], )
    code_rect = cv2.minAreaRect(code_points)
    red_center = [np.mean(red_points[:, 0]), np.mean(red_points[:, 1])]
    code_center = [np.mean(code_points[:, 0]), np.mean(code_points[:, 1])]
    if code_center[0] == red_center[0]:
        angle = 0 if red_center[1] < code_center[1] else -180
    else:
        slope_rate = (code_center[1] - red_center[1]) / (code_center[0] - red_center[0])
        if red_center[1] < code_center[1]:
            angle = math.atan(slope_rate) / math.pi * 180
            angle = angle - 90 if angle >= 0 else 90 + angle
        elif red_center[1] > code_center[1]:
            angle = math.atan(slope_rate) / math.pi * 180
            angle = -(90 - angle) if angle < 0 else -(270 - angle)
        else:
            angle = 90 if code_center[0] < red_center[0] else -90
    # center_cord = tuple(map(round, code_center))
    w, h = max(code_rect[1]), min(code_rect[1])
    M = cv2.getRotationMatrix2D(tuple(code_center), angle, 1.0)
    # rotate_img = cv2.warpAffine(img, M, tuple(map(int, code_rect[1])), flags=cv2.WARP_INVERSE_MAP,
    rotate_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if debug:
        cv2.imwrite('rotate_img.png', rotate_img)
    warped_img = rotate_img[int(round(code_center[1] - h / 2 + 2)): int(round(code_center[1] + h / 2 - 1)),
                            int(round(code_center[0] - w / 2)): int(round(code_center[0] + w / 2)), :]
    if debug:
        cv2.imwrite('warped_code_img.png', warped_img)
    code_width = warped_img.shape[1] / 5

    x5 = warped_img[:, :round(code_width), :]
    x4 = warped_img[:, round(code_width):round(code_width * 2), :]
    x3 = warped_img[:, round(code_width * 2):round(code_width * 3), :]
    x2 = warped_img[:, round(code_width * 3):round(code_width * 4), :]
    x1 = warped_img[:, round(code_width * 4):round(code_width * 5), :]

    x5 = cvt_img_2_num(x5)
    x4 = cvt_img_2_num(x4)
    x3 = cvt_img_2_num(x3)
    x2 = cvt_img_2_num(x2)
    x1 = cvt_img_2_num(x1)

    ctl_code = x5 * pow(4, 4) + x4 * pow(4, 3) + x3 * pow(4, 2) + x2 * pow(4, 1) + x1
    # print(time.time())
    cv2.drawContours(img, [code_points], -1, (255, 255, 0), 1)
    cv2.drawContours(img, [red_points], -1, (255, 255, 0), 1)
    cv2.drawContours(img, [white_box.astype(np.int)], -1, (0, 255, 0), 1)
    cv2.circle(img, (round(ctl_point[0]), round(ctl_point[1])), 1, (0, 0, 255))
    cv2.putText(img, str(ctl_code), (int(ctl_point[0]) + 20, int(ctl_point[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (0, 0, 255), 1)
    # cv2.imwrite('possible.png', img)

    # code_rect = cv2.minAreaRect(code_points)

    # print(white_box)
    # print(outer_box)
    return img


def contour_find(img_file):
    # time0 = time.time()
    # gray_img = cv2.imread(img_file, 0)
    orig_img = cv2.imread(img_file)
    img = np.copy(orig_img)
    img[:, :, 2] = np.clip(orig_img[:, :, 2], a_min=0, a_max=130)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.blur(gray_img, (3, 3))
    # cv2.imwrite('gray_img.jpg', gray_img)
    # dst_img = cv2.equalizeHist(gray_img)
    # cv2.imwrite('equa_hist.jpg', dst_img)
    # norm_img = np.zeros_like(gray_img)
    # cv2.normalize(gray_img, norm_img, 0, 128, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite('norm_img.jpg', norm_img)
    time1 = time.time()
    _, thresh_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imwrite('thresh_img.png', thresh_img)

    # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 3)
    # 自适应阈值二值化速度太慢了，无法满足要求
    #
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(time.time() - time1)
    # for cnt in contours:
    hierarchy = np.squeeze(hierarchy)
    ic, parent_idx = 0, -1
    rects = []
    cnts = []
    if debug:
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        cv2.imwrite('contours0.png', img)
    for i in range(len(contours)):
        if hierarchy[i][2] != -1 and ic == 0:
            parent_idx = i
            ic += 1
        elif hierarchy[i][2] != -1:
            ic += 1
        elif hierarchy[i][2] == -1:
            ic, parent_idx = 0, -1

        if ic >= 2:
            # parent_idx = hierarchy[i][3]
            parent_cnt = contours[parent_idx]
            # child_cnt = contours[i]
            ic, parent_idx = 0, -1

            if not contour_chk(parent_cnt, thresh_img):
                continue
            # contour_chk(parent_cnt, gray_img)
            rect = cv2.minAreaRect(parent_cnt)

            # print('contour likely:', d1)
            rects.append(rect)
            cnts.append(parent_cnt)

    print(time.time() - time1)
    if len(rects) != 1:
        # todo
        print("ctl point count:", len(rects))
        print("can't find any ctl point or more than one ctl point")
        # return
    # time1 = time.time()
    # print(time.time() - time0)
    # pts = cv2.boxPoints(rects[0]).astype(np.int)
    # img = cv2.imread(img_file)
    # cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    # img = cv2.imread(img_file)
    for rect in rects:
        orig_img = crop_possible_region(rect, orig_img)
    print(time.time() - time1)
    cv2.imwrite(os.path.split(img_file)[-1], orig_img)

    print('aaaaa')


if __name__ == '__main__':
    # hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    # test_img = r'/media/chen/wt/tmp/control_point/0929-2_D_0305_code_a3.jpg'
    test_img = r'/media/chen/wt/tmp/control_point/0929-2_D_0826_code_a3_v2.jpg'
    kh_img_file_path = '/media/chen/wt/tmp/kk/kh.png'
    kgt_img_file_path = '/media/chen/wt/tmp/kk/sdyf_kh.png'
    # test_img = r'/media/chen/wt/tmp/control_point/0929-2_D_0856_2.jpg'
    test_dir = 'E:\\wt\\ctl_point\\test_images\\'
    test_imgs = glob.glob(test_dir + '*.jpg')
    debug = True
    for img in test_imgs:
        img = r'E:\\wt\\ctl_point\\test_images\\test-B-0366.JPG'
        contour_find(img)
    # locate_ctl_point(test_img)
    # kgt_area_detect(test_img)
    # training_data_path = r'/media/chen/wt/tmp/kk/data'
    # yolo_data = r'/media/chen/wt/tmp/kk/yolo_data'
    # # generate_data(training_data_path, 100)
    # # label_file = os.path.join(training_data_path, 'label.txt')
    # # kgt_area_detect(test_img)
    # # generate_reality_data('', training_data_path, 'kh')
    # # split_reality_data(kgt_img_file_path, r'/media/chen/wt/tmp/kk/sdyf_kh.txt', training_data_path, 'shyf_kh')
    # # infer(yolo_data)
    # label_with_template(yolo_data, '1085253_kh_01085253_10239_201809282_183.txt')
    # # crop_orig_paper('1082844', [691, 432, 1528, 894], [242, 1016, 911, 1163], '/media/chen/wt/tmp/kk/yolo_data')
    # # mnist_test()
