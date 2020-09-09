import PIL.ImageFont as ImageFont
from mk_ancient_poems_data.ocr_interface import test_single_image
from mk_ancient_poems_data.gushici_cut import get_mask_img, get_full_score_img, gushici_cut
import os


if __name__ == "__main__":
    # test_single_image()
    id_root_dir = '/media/chen/wt/download_paper/paper/'
    block_rec_dir = '/media/chen/wt/data/hw_poem_data/block_rec'
    gushici_rec_dir = '/media/chen/wt/data/hw_poem_data/gushici_rec'
    poem_test_dir = '/media/chen/wt/data/hw_poem_data/poem_test'
    dealed_txt = os.path.join(block_rec_dir, 'tmp.txt')
    get_mask_img(id_root_dir, block_rec_dir, dealed_txt)
    get_full_score_img(id_root_dir, block_rec_dir, gushici_rec_dir, poem_test_dir)
