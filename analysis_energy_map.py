import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from util import html

cyclegan_path = 'results/brats_flair_t1_rawcyclegan_autoaddtumor/test_latest/images/'
energy_path = 'results/brats_flair_t1_energy_autoaddtumor/test_latest/images/'
out_path = 'results/analysis_heatmap/'
try_cell_range = 16

def get_heatmap(cell_range, img_in, img_out):
    img_in_mat = np.array(img_in, dtype=np.float)
    img_out_mat = np.array(img_out, dtype=np.float)
    img_in_prob = img_in_mat /127.0 -1
    img_out_prob = img_out_mat /127.0 -1
    mat_index = np.int32(img_in_mat / cell_range) 
    mat_feature = img_out_prob - img_in_prob
    bins_num = 256 / cell_range
    bins_idx = np.zeros(bins_num)
    for idx in range(bins_num):
        num = np.sum(mat_index == idx)
        if num >0:
            bins_idx[idx] = np.mean(mat_feature[mat_index == idx])
    return bins_idx

def draw_webpage_gt(img_name, out_path):
    real_A = Image.open(energy_path + img_name + '_real_A.png')
    real_B = Image.open(energy_path + img_name + '_real_B.png')
    real_A.save(out_path + img_name + '_gt_real_A.png')
    real_B.save(out_path + img_name + '_gt_real_B.png')
    heatmapAB = get_heatmap(try_cell_range, real_A, real_B)
    heatmapBA = get_heatmap(try_cell_range, real_B, real_A)
    plt.plot(heatmapAB)
    plt.savefig(out_path + img_name + '_gt_heatmap_AB.png')
    plt.plot(heatmapBA)
    plt.savefig(out_path + img_name + '_gt_heatmap_BA.png')
    ims = [img_name + '_gt_real_A.png', img_name + '_gt_real_B.png', img_name + '_gt_heatmap_AB.png', img_name + '_gt_heatmap_BA.png']
    txts = ['gt_real_A.png', 'gt_real_B', 'gt_heatmap_AB', 'gt_heatmap_BA']
    links = [img_name + '_gt_real_A.png', img_name + '_gt_real_B.png', img_name + '_gt_heatmap_AB.png', img_name + '_gt_heatmap_BA.png']
    return ims, txts, links

def draw_webpage_cyclegan(img_name, in_path, out_path):
    real_A = Image.open(in_path + img_name + '_real_A.png')
    fake_B = Image.open(in_path + img_name + '_fake_B.png')
    rec_A = Image.open(in_path + img_name + '_rec_A.png')
    real_B = Image.open(in_path + img_name + '_real_B.png')
    fake_A = Image.open(in_path + img_name + '_fake_A.png')
    rec_B = Image.open(in_path + img_name + '_rec_B.png')
    real_A.save(out_path + img_name + '_cyclegan_real_A.png')
    fake_B.save(out_path + img_name + '_cyclegan_fake_B.png')
    rec_A.save(out_path + img_name + '_cyclegan_rec_A.png')
    real_B.save(out_path + img_name + '_cyclegan_real_B.png')
    fake_A.save(out_path + img_name + '_cyclegan_fake_A.png')
    rec_B.save(out_path + img_name + '_cyclegan_rec_B.png')
    heatmapAB = get_heatmap(try_cell_range, real_A, fake_B) 
    heatmapBA = get_heatmap(try_cell_range, real_B, fake_A) 
    plt.plot(heatmapAB)
    plt.savefig(out_path + img_name + '_cyclegan_heatmap_AB.png')
    plt.plot(heatmapBA)
    plt.savefig(out_path + img_name + '_cyclegan_heatmap_BA.png')
    ims = [img_name + '_cyclegan_real_A.png', img_name + '_cyclegan_fake_B.png', img_name + '_cyclegan_rec_A.png', img_name + '_cyclegan_real_B.png', img_name + '_cyclegan_fake_A.png', img_name + '_cyclegan_rec_B.png']
    txts = ['cyclegan_real_A', 'cyclegan_fake_B', 'cyclegan_rec_A', 'cyclegan_real_B', 'cyclegan_fake_A', 'cyclegan_rec_B.']
    links = [img_name + '_cyclegan_real_A.png', img_name + '_cyclegan_fake_B.png', img_name + '_cyclegan_rec_A.png', img_name + '_cyclegan_real_B.png', img_name + '_cyclegan_fake_A.png', img_name + '_cyclegan_rec_B.png']
    return ims, txts, links

web_dir = out_path
webpage = html.HTML(web_dir, 'analysis_heatmap')
for pre in ['HG', 'LG']:
    for idx in range(1,25+1):
        img_name = '%s_%04d_090' %(pre, idx)
        webpage.add_header(img_name + '_groundtruth')
        ims, txts, links = draw_webpage_gt(img_name, out_path)
        webpage.add_images(ims, txts, links, width=256)
        ims, txts, links = draw_webpage_cyclegan(img_name, cyclegan_path, out_path)
        webpage.add_images(ims, txts, links, width=256)
        ims, txts, links = draw_webpage_cyclegan(img_name, energy_path, out_path)
        webpage.add_images(ims, txts, links, width=256) 
web_page.save()
