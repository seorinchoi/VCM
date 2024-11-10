import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

def evaluation(datanum, xmin, ymin, xmax, ymax):

    image_dir = "/content/drive/MyDrive/SMC/test/images"
    gt_dir = "/content/drive/MyDrive/SMC/test/labels"
    pred_dir = "/content/drive/MyDrive/work_dirs/vit/b16lr001/format_result"
    result_dir = "/content/drive/MyDrive/work_dirs/vit/b16lr001/format_result/eval_results"
    resultimg_dir = "/content/drive/MyDrive/work_dirs/vit/b16lr001/format_result/eval_results/img"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(resultimg_dir):
        os.makedirs(resultimg_dir)

    pred_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir)]
    gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    img_names = []
    accus, sens, specs, dscs, ppvs, npvs = [], [], [], [], [], []
    tp_a, fp_a, fn_a, tn_a = [], [], [], []

    for pred_file, gt_file, image_file in zip(pred_files, gt_files, image_files):

        print("Extracting ground truth bboxes for set {}...".format(pred_file))
        
        # 읽어온 마스크와 이미지
        pred_mask = plt.imread(pred_file)
        if pred_mask.ndim > 2:
            pred_mask = pred_mask[:, :, 0]
        pred_mask = np.float32(pred_mask)

        gt_mask = plt.imread(gt_file)
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = np.float32(gt_mask)

        img = cv2.imread(image_file)
        if img.ndim > 2:
            img1 = img[:, :, 0]
            img1 = np.float32(img1)

        # 실제 사용될 xmax, ymax를 배열 크기로 제한
        mask_height, mask_width = gt_mask.shape
        xmax_adj = min(xmax, mask_width)
        ymax_adj = min(ymax, mask_height)

        tp = fp = fn = tn = 0
        img_a_mask = np.zeros_like(img, dtype=np.uint8)

        for y in range(ymin, ymax_adj):
            for x in range(xmin, xmax_adj):
                if (gt_mask[y, x] != 0) and (pred_mask[y, x] != 0):
                    tp += 1
                    img_a_mask[y, x, 2] = 255
                elif (gt_mask[y, x] == 0) and (pred_mask[y, x] != 0):
                    fp += 1
                    img_a_mask[y, x, 0] = 255
                elif (gt_mask[y, x] != 0) and (pred_mask[y, x] == 0):
                    fn += 1
                    img_a_mask[y, x, 1] = 255
                elif (gt_mask[y, x] == 0) and (pred_mask[y, x] == 0):
                    tn += 1

        final_gt_mask = cv2.addWeighted(img, 1.0, img_a_mask, 0.5, 0)
        outfile = os.path.join(resultimg_dir, str(os.path.basename(gt_file)))
        cv2.imwrite(outfile, final_gt_mask)

        # 정확도, 민감도 등 계산
        accus.append((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0)
        sens.append(tp / (tp + fn) if (tp + fn) else 0)
        specs.append(tn / (tn + fp) if (tn + fp) else 0)
        dscs.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0)
        ppvs.append(tp / (tp + fp) if (tp + fp) else 0)
        npvs.append(tn / (tn + fn) if (tn + fn) else 0)

        tp_a.append(tp)
        fp_a.append(fp)
        fn_a.append(fn)
        tn_a.append(tn)

    df = pd.DataFrame({
        'img': [os.path.basename(f) for f in gt_files],
        'TP': tp_a, 'FP': fp_a, 'FN': fn_a, 'TN': tn_a,
        'Accu': accus, 'Sen': sens, 'Spec': specs, 'DSC': dscs, 'PPV': ppvs, 'NPV': npvs
    })
    outfile = os.path.join(result_dir, 'evaluation.csv')
    df.to_csv(outfile, index=False)

if __name__ == "__main__":
    data_test_a = [1]
    for datanum in data_test_a:
        evaluation(datanum, xmin=0, ymin=0, xmax=256, ymax=256)
