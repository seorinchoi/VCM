''' test_eval.py - test evaluation with eta '''
import cv2
import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import time

# Directory Settings
image_dir = "./SMC/Meniscus/test/images"
gt_dir = "./SMC/Meniscus/test/labels"
pred_dir =  './SMC/Segmenter-swin/patch2/Focal/batch16/format_results'
result_dir = "./SMC/Segmenter-swin/patch2/Focal/batch16/eval_results"
resultimg_dir = "./SMC/Segmenter-swin/patch2/Focal/batch16/eval_results/img"


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def calculate_metrics(image_file, gt_file, pred_file, xmin, ymin, xmax, ymax):
    # Load the images
    image = cv2.imread(image_file)
    gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)

    # Check if masks are loaded correctly
    if gt_mask is None:
        print(f"Failed to load ground truth mask from {gt_file}")
    if pred_mask is None:
        print(f"Failed to load prediction mask from {pred_file}")

    # Convert the masks to float32
    gt_mask = np.float32(gt_mask)
    pred_mask = np.float32(pred_mask)

    tp = fp = fn = tn = 0

    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            if (gt_mask[y, x] != 0) and (pred_mask[y, x] != 0):
                tp += 1
            elif (gt_mask[y, x] == 0) and (pred_mask[y, x] != 0):
                fp += 1
            elif (gt_mask[y, x] != 0) and (pred_mask[y, x] == 0):
                fn += 1
            elif (gt_mask[y, x] == 0) and (pred_mask[y, x] == 0):
                tn += 1

    return tp, fp, fn, tn

def visualize_and_save(image_file, gt_file, pred_file, resultimg_dir):
    image = cv2.imread(image_file)
    gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)

    if gt_mask is None or pred_mask is None:
        print(f"Failed to load masks for {image_file}")
        return

    img_name = os.path.basename(image_file)
    img_a_mask = np.zeros_like(image, dtype=np.uint8)

    for y in range(gt_mask.shape[0]):
        for x in range(gt_mask.shape[1]):
            if (gt_mask[y, x] != 0) and (pred_mask[y, x] != 0):
                img_a_mask[y, x, 2] = 255
            if (gt_mask[y, x] == 0) and (pred_mask[y, x] != 0):
                img_a_mask[y, x, 0] = 255
            if (gt_mask[y, x] != 0) and (pred_mask[y, x] == 0):
                img_a_mask[y, x, 1] = 255

    final_gt_mask = cv2.addWeighted(image, 1.0, img_a_mask, 0.5, 0)
    outfile = os.path.join(resultimg_dir, str(img_name))
    cv2.imwrite(outfile, final_gt_mask)
    print(f"Saved visualization to {outfile}")

def get_sorted_file_list(directory):
    file_list = os.listdir(directory)
    return sorted(file_list, key=natural_sort_key)

def evaluation(xmin, ymin, xmax, ymax):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(resultimg_dir):
        os.makedirs(resultimg_dir)

    # Get and sort file names using natural sort
    image_files = get_sorted_file_list(image_dir)
    gt_files = get_sorted_file_list(gt_dir)
    pred_files = get_sorted_file_list(pred_dir)

    img_names = []
    tp_a = []
    fp_a = []
    fn_a = []
    tn_a = []

    total_files = len(image_files)
    start_time = time.time()

    for idx, (img_name, gt_name, pred_name) in enumerate(zip(image_files, gt_files, pred_files)):
        image_file = os.path.join(image_dir, img_name)
        gt_file = os.path.join(gt_dir, gt_name)
        pred_file = os.path.join(pred_dir, pred_name)

        print(f"Processing file: {img_name}")
        tp, fp, fn, tn = calculate_metrics(image_file, gt_file, pred_file, xmin, ymin, xmax, ymax)
        visualize_and_save(image_file, gt_file, pred_file, resultimg_dir)

        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / (idx + 1)
        eta = avg_time_per_file * (total_files - (idx + 1))

        # Convert seconds to hours, minutes, and seconds
        hours, rem = divmod(eta, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f'{img_name}: TP : {tp}, FP : {fp}, FN : {fn}, TN : {tn}')
        print(f"ETA: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds remaining")


        img_names.append(img_name)
        tp_a.append(tp)
        fp_a.append(fp)
        fn_a.append(fn)
        tn_a.append(tn)

    df = pd.DataFrame({
        'img': img_names,
        'TP': tp_a,
        'FP': fp_a,
        'FN': fn_a,
        'TN': tn_a,
    })
    outfile = os.path.join(result_dir, 'evaluation.csv')
    df.to_csv(outfile, index=False)

if __name__ == "__main__":
    xmin = 0
    ymin = 0
    xmax = 291
    ymax = 80

    evaluation(xmin, ymin, xmax, ymax)
