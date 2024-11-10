# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:39:59 2018

@author: VCMI
"""# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:39:59 2018

@author: VCMI
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

def evaluation(datanum, xmin, ymin, xmax, ymax):

    image_dir = "./datasets/test-small/images/"
    gt_dir = "./datasets/test-small/masks/"
    pred_dir = "./work_dirs/Swin-Seg/batch24lr0.0005/format_results"
    result_dir = "./work_dirs/Swin-Seg/batch24lr0.0005/eval_results"
    resultimg_dir = "./work_dirs/Swin-Seg/batch24lr0.0005/eval_results/img"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(resultimg_dir):
        os.makedirs(resultimg_dir)

    pred_files = os.listdir(pred_dir)
    pred_files = [os.path.join(pred_dir, f) for f in pred_files]

    gt_files = os.listdir(gt_dir)
    gt_files = [os.path.join(gt_dir, f) for f in gt_files]

    image_files = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, f) for f in image_files]

    img_names = []
    accus = []
    sens = []
    specs = []
    dscs = []
    ppvs = []
    npvs = []
    tp_a = []
    fp_a = []
    fn_a = []
    tn_a = []

    tp_lowROI_a = []
    fp_lowROI_a = []
    fn_lowROI_a = []
    tn_lowROI_a = []

  # for auto_file, mask_file, low_ROI_file, image_file in zip(auto_files, mask_files, low_ROI_files,  image_files):
    for pred_file, gt_file, image_file in zip(pred_files, gt_files, image_files):

        # SB ROI 설정
        xmin = 0
        ymin = 0
        xmax = 255
        ymax = 255

        print("Extracting ground truth bboxes for set {}...".format(pred_file))
        tp = fp = fn = tn = 0
        tp_lowROI = fp_lowROI = fn_lowROI = tn_lowROI = 0
        non_lowROI = 0
        img_name = os.path.basename(gt_file)
        img_names.append(img_name)

        pred_mask = plt.imread(pred_file)
        # _, pred_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
        if pred_mask.ndim > 2:
            pred_mask = pred_mask[:, :, 0]
        pred_mask = np.float32(pred_mask)

        img = cv2.imread(image_file)
        if img.ndim > 2:
            img1 = img[:, :, 0]
            img1 = np.float32(img1)

        gt_mask = plt.imread(gt_file)
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = np.float32(gt_mask)

        img_a_mask = np.zeros_like(img, dtype=np.uint8)
        # img_only_gt_mask = np.zeros_like(img, dtype=np.uint8)
        img_a_lowROI_mask = np.zeros_like(img, dtype=np.uint8)

        for y in range(ymin, ymax):
            for x in range(int(xmin), xmax):
                # if (img1[y, x] <= 212.0):   #low일때만
                if (gt_mask[y, x] != 0) and (pred_mask[y, x] != 0):
                    tp = tp + 1
                    img_a_mask[y, x, 2] = 255
                if (gt_mask[y, x] == 0) and (pred_mask[y, x] != 0):
                    fp = fp + 1
                    img_a_mask[y, x, 0] = 255
                if (gt_mask[y, x] != 0) and (pred_mask[y, x] == 0):
                    # img_only_gt_mask[y, x, 1] = 255
                    img_a_mask[y, x, 1] = 255
                    fn = fn + 1
                if (gt_mask[y, x] == 0) and (pred_mask[y, x] == 0):
                    tn = tn + 1

        final_gt_mask = cv2.addWeighted(img, 1.0, img_a_mask, 0.5, 0)
        try:
            outfile = os.path.join(resultimg_dir, str(img_name))
        except FileNotFoundError:
            os.mkdir(resultimg_dir)
            outfile = os.path.join(resultimg_dir, str(img_name))

        cv2.imwrite(outfile, final_gt_mask)

        # final_only_gt_mask = cv2.addWeighted(img, 0.5, img_only_gt_mask, 0.5, 0)
        # outfile = os.path.join(result_dir, 'only_' + str(img_name))
        # cv2.imwrite(outfile, final_only_gt_mask)

        # accuracy
        if (tp + tn + fp + fn == 0):
            accu = 0
        else:
            accu = float((tp + tn) / (tp + tn + fp + fn))
        accus.append(accu)

        # sensitivity
        if (tp + fn == 0):
            sen = 0
        else:
            sen = float(tp / (tp + fn))
        sens.append(sen)

        # specificity
        if (tn + fp == 0):
            spec = 0
        else:
            spec = float(tn / (tn + fp))
        specs.append(spec)

        # DSC
        if (tp + fp + tp + fn == 0):
            dsc = 0
        else:
            dsc = float(2 * tp / (tp + fp + tp + fn))
        dscs.append(dsc)

        # PPV
        if (tp + fp == 0):
            ppv = 0
        else:
            ppv = float(tp / (tp + fp))
        ppvs.append(ppv)

        # NPV
        if (tn + fn == 0):
            npv = 0
        else:
            npv = float(tn / (tn + fn))
        npvs.append(npv)

        tp_a.append(tp)
        fp_a.append(fp)
        fn_a.append(fn)
        tn_a.append(tn)

    # df = {'img': img_names, 'TP': tp_a, 'FP': fp_a, 'FN': fn_a, 'TN': tn_a,
    #       'Accu': accus, 'Sen': sens, 'Spec': specs, 'DSC': dscs, 'PPV': ppvs, 'NPV': npvs}
    df = {'img': img_names, 'TP': tp_a, 'FP': fp_a, 'FN': fn_a, 'TN': tn_a}
    df = pd.DataFrame(df)
    # if (number == 0):
    #     outfile = os.path.join(result_dir, 'evaluation_{}_axl.csv'.format(str(datanum)))
    # elif (number == 1):
    #     outfile = os.path.join(result_dir, 'evaluation_{}_cor.csv'.format(str(datanum)))
    # elif (number == 2):
    #     outfile = os.path.join(result_dir, 'evaluation_{}_sag.csv'.format(str(datanum)))

    outfile = os.path.join(result_dir, 'evaluation.csv')
    df.to_csv(outfile)



if __name__ == "__main__":

#    data_test_a = [0, 1, 2, 3, 4]
#     data_test_a = [9, 19, 42, 51, 60, 69, 80, 91]
#     data_test_a = [9, 18, 27, 36, 44, 51]
    data_test_a = [1]

    for datanum in data_test_a:
        # for dataplane in data_plane:

            xmin = 0
            ymin = 0
            xmax = 256
            ymax = 256

            evaluation(datanum, xmin, ymin, xmax, ymax)
