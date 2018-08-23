import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from util import html

cyclegan_path = 'results/brats_flair_t1_rawcyclegan_autoaddtumor/test_latest/images/'
energy_path = 'results/brats_flair_t1_energy_autoaddtumor/test_latest/images/'
out_root = 'results/analysis_heatmap/'
out_path = 'results/analysis_heatmap/images/'
try_cell_range = 16

def get_heatmap(cell_range, img_in, img_out):
    img_in_mat = np.array(img_in, dtype=np.float)
    img_out_mat = np.array(img_out, dtype=np.float)
    img_in_prob = img_in_mat /127.0 -1
    img_out_prob = img_out_mat /127.0 -1
    mat_index = np.int32(img_in_mat / cell_range) 
    mat_feature = img_out_prob - img_in_prob
    bins_num = 256 / cell_range
    bins_mean = np.zeros(bins_num)
    bins_var = np.zeros(bins_num)
    for idx in range(bins_num):
        num = np.sum(mat_index == idx)
        if num >0:
            bins_mean[idx] = np.mean(mat_feature[mat_index == idx])
            bins_var[idx] = np.std(mat_feature[mat_index == idx])
    return bins_mean, bins_var

def get_minmax(cell_range, img_in, img_out):
    img_in_mat = np.array(img_in, dtype=np.float)
    img_out_mat = np.array(img_out, dtype=np.float)
    img_in_prob = img_in_mat /127.0 -1
    img_out_prob = img_out_mat /127.0 -1
    mat_index = np.int32(img_in_mat / cell_range) 
    mat_feature = img_out_prob - img_in_prob
    bins_num = 256 / cell_range
    bins_max = np.zeros(bins_num)
    bins_min = np.zeros(bins_num)
    for idx in range(bins_num):
        num = np.sum(mat_index == idx)
        if num >0:
            prob_mat = mat_feature[mat_index == idx]
            bins_max[idx] = prob_mat.max()
            bins_min[idx] = prob_mat.min()
    return bins_min, bins_max

def get_RGB_fusion(img_in, img_out):
    img_in_mat = np.array(img_in)
    img_out_mat = np.array(img_out)
    img_RGB = np.zeros([img_in_mat.shape[0], img_in_mat.shape[1], 3], dtype=np.uint8)
    img_RGB[:,:,0] = img_in_mat[:,:,0]
    img_RGB[:,:,1] = img_out_mat[:,:,0]
    img_res = Image.fromarray(img_RGB)
    return img_res

def draw_webpage_gt(img_name, out_path):
    real_A = Image.open(energy_path + img_name + '_real_A.png')
    real_B = Image.open(energy_path + img_name + '_real_B.png')
    real_A.save(out_path + img_name + '_gt_real_A.png')
    real_B.save(out_path + img_name + '_gt_real_B.png')
    heatmap_mean_AB, heatmap_var_AB = get_heatmap(try_cell_range, real_A, real_B)
    bins_min_AB, bins_max_AB = get_minmax(try_cell_range, real_A, real_B)
    heatmap_mean_BA, heatmap_var_BA = get_heatmap(try_cell_range, real_B, real_A)
    bins_min_BA, bins_max_BA = get_minmax(try_cell_range, real_B, real_A)
    #plt.plot(heatmapAB)
    #plt.errorbar(range(try_cell_range), heatmap_mean_AB, yerr=[heatmap_mean_AB-bins_min_AB, bins_max_AB-heatmap_mean_AB], fmt='--o')
    plt.errorbar(range(try_cell_range), heatmap_mean_AB, yerr=[heatmap_var_AB, heatmap_var_AB], fmt='--o')
    plt.axis([-1, 16, -2.0, 2.0])
    plt.savefig(out_path + img_name + '_gt_heatmap_AB.png')
    plt.close()
    #plt.plot(heatmapBA)
    #plt.errorbar(range(try_cell_range), heatmap_mean_BA, yerr=[heatmap_mean_BA-bins_min_BA, bins_max_BA-heatmap_mean_BA], fmt='--o')
    plt.errorbar(range(try_cell_range), heatmap_mean_BA, yerr=[heatmap_var_BA, heatmap_var_BA], fmt='--o')
    plt.axis([-1, 16, -2.0, 2.0])
    plt.savefig(out_path + img_name + '_gt_heatmap_BA.png')
    plt.close()
    img_rgb = get_RGB_fusion(real_A, real_B)
    img_rgb.save(out_path + img_name + '_gt_RGB.png')
    ims = [img_name + '_gt_real_A.png', img_name + '_gt_real_B.png', img_name + '_gt_RGB.png', img_name + '_gt_heatmap_AB.png', img_name + '_gt_heatmap_BA.png']
    txts = ['gt_real_A.png', 'gt_real_B', 'gt_RGB', 'gt_heatmap_AB', 'gt_heatmap_BA']
    links = [img_name + '_gt_real_A.png', img_name + '_gt_real_B.png', img_name + '_gt_RGB.png', img_name + '_gt_heatmap_AB.png', img_name + '_gt_heatmap_BA.png']
    return ims, txts, links

def draw_webpage_cyclegan(img_name, in_path, out_path, suffix):
    real_A = Image.open(in_path + img_name + '_real_A.png')
    fake_B = Image.open(in_path + img_name + '_fake_B.png')
    rec_A = Image.open(in_path + img_name + '_rec_A.png')
    real_B = Image.open(in_path + img_name + '_real_B.png')
    fake_A = Image.open(in_path + img_name + '_fake_A.png')
    rec_B = Image.open(in_path + img_name + '_rec_B.png')
    real_A.save(out_path + img_name + suffix + '_real_A.png')
    fake_B.save(out_path + img_name + suffix + '_fake_B.png')
    rec_A.save(out_path + img_name + suffix + '_rec_A.png')
    real_B.save(out_path + img_name + suffix + '_real_B.png')
    fake_A.save(out_path + img_name + suffix + '_fake_A.png')
    rec_B.save(out_path + img_name + suffix + '_rec_B.png')
    heatmap_mean_AB, heatmap_var_AB = get_heatmap(try_cell_range, real_A, fake_B) 
    heatmap_mean_AB_rec, heatmap_var_AB_rec = get_heatmap(try_cell_range, fake_B, real_A)
    heatmap_mean_BA, heatmap_var_BA = get_heatmap(try_cell_range, real_B, fake_A) 
    heatmap_mean_BA_rec, heatmap_var_BA_rec = get_heatmap(try_cell_range, fake_A, real_B)
    bins_min_AB, bins_max_AB = get_minmax(try_cell_range, real_A, fake_B)
    bins_min_ABrec, bins_max_ABrec = get_minmax(try_cell_range,fake_B, real_A)
    bins_min_BA, bins_max_BA = get_minmax(try_cell_range, real_B, fake_A)
    bins_min_BArec, bins_max_BArec = get_minmax(try_cell_range,fake_A, real_B)
    #plt.plot(heatmapAB)
    #plt.errorbar(range(try_cell_range), heatmap_mean_AB, yerr=[heatmap_mean_AB-bins_min_AB, bins_max_AB-heatmap_mean_AB], fmt='--o')
    plt.errorbar(range(try_cell_range), heatmap_mean_AB, yerr=[heatmap_var_AB, heatmap_var_AB], fmt='--o')
    plt.axis([-1, 16, -2.0, 2.0])
    plt.savefig(out_path + img_name + suffix + '_heatmap_AB.png')
    plt.close()
    #plt.errorbar(range(try_cell_range), heatmapAB_rec, yerr=[heatmapAB_rec-bins_min_ABrec, bins_max_ABrec-heatmapAB_rec], fmt='--o')
    plt.errorbar(range(try_cell_range), heatmap_mean_AB_rec, yerr=[heatmap_var_AB_rec, heatmap_var_AB_rec], fmt='--o')
    plt.axis([-1, 16, -2.0, 2.0])
    plt.savefig(out_path + img_name + suffix + '_heatmap_ABrec.png')
    plt.close()
    #plt.plot(heatmapBA)
    #plt.errorbar(range(try_cell_range), heatmapBA, yerr=[heatmapBA-bins_min_BA, bins_max_BA-heatmapBA], fmt='--o')
    plt.errorbar(range(try_cell_range), heatmap_mean_BA, yerr=[heatmap_var_BA, heatmap_var_BA], fmt='--o')
    plt.axis([-1, 16, -2.0, 2.0])
    plt.savefig(out_path + img_name + suffix + '_heatmap_BA.png')
    plt.close()
    #plt.errorbar(range(try_cell_range), heatmapBA_rec, yerr=[heatmapBA_rec-bins_min_BArec, bins_max_BArec-heatmapBA_rec], fmt='--o')
    plt.errorbar(range(try_cell_range), heatmap_mean_BA_rec, yerr=[heatmap_var_BA_rec, heatmap_var_BA_rec], fmt='--o')
    plt.axis([-1, 16, -2.0, 2.0])
    plt.savefig(out_path + img_name + suffix + '_heatmap_BArec.png')
    plt.close()
    ims = [img_name + suffix + '_real_A.png', img_name + suffix + '_fake_B.png', img_name + suffix + '_rec_A.png', 
           img_name + suffix + '_heatmap_AB.png', img_name + suffix + '_heatmap_ABrec.png',
           img_name + suffix + '_real_B.png', img_name + suffix + '_fake_A.png', img_name + suffix + '_rec_B.png', 
           img_name + suffix + '_heatmap_BA.png', img_name + suffix + '_heatmap_BArec.png']
    txts = [suffix + '_real_A', suffix + '_fake_B', suffix + '_rec_A', 
            suffix + '_heatmap_AB', suffix + '_heatmap_ABrec',
            suffix + '_real_B', suffix + '_fake_A', suffix + '_rec_B', 
            suffix + '_heatmap_BA', suffix + '_heatmap_BArec']
    links = [img_name + suffix + '_real_A.png', img_name + suffix + '_fake_B.png', img_name + suffix + '_rec_A.png',
             img_name + suffix + '_heatmap_AB.png', img_name + suffix + '_heatmap_ABrec.png',
             img_name + suffix + '_real_B.png', img_name + suffix + '_fake_A.png', img_name + suffix + '_rec_B.png', 
             img_name + suffix + '_heatmap_BA.png', img_name + suffix + '_heatmap_BArec.png']
    return ims, txts, links

web_dir = out_root
webpage = html.HTML(web_dir, 'analysis_heatmap')
for pre in ['HG', 'LG']:
    for idx in range(1,25+1):
        img_name = '%s_%04d_090' %(pre, idx)
        print img_name
        webpage.add_header(img_name + '_groundtruth')
        ims, txts, links = draw_webpage_gt(img_name, out_path)
        webpage.add_images(ims, txts, links, width=256)
        ims, txts, links = draw_webpage_cyclegan(img_name, cyclegan_path, out_path, '_cyclegan')
        webpage.add_images(ims, txts, links, width=256)
        ims, txts, links = draw_webpage_cyclegan(img_name, energy_path, out_path, '_energy')
        webpage.add_images(ims, txts, links, width=256) 
webpage.save()
