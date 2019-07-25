import os
import click
import numpy as np

def compute_IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

n_TP = 0
n_FP = 0
n_FN = 0

def calc_points(IoU):
    if IoU >= 0.6:
        return IoU * 3
    return (1 - IoU) * (-1)
    
def get_score(sorted_pairs, len_pred):
    global n_TP, n_FP, n_FN;
    score = 0
    temp = []
    temp_GT = []
    temp_D = []
    
    for k, v in sorted_pairs:
        if v > 0:
            gt, d = k.split('_')
            if gt not in temp_GT and d not in temp_D:
                temp.append(k); temp_GT.append(gt); temp_D.append(d)
                score += calc_points(v)
                n_TP += 1
                
    n_FN += len_pred - len(temp)
    score += (-1) * (len_pred - len(temp))
    
    return score

fn = lambda x: float(x.strip())

@click.command()
@click.argument('y_true', type=click.File('r'))
@click.argument('y_scores', type=click.File('r'))
@click.option('--threshold', type=click.FloatRange(0, 1), default=0.6)
def main(y_true, y_scores, threshold):
    gt_values = {}
    for i in y_true:
        key = i.split(',')[0].strip()
        values = list(map(fn, i.split(',')[1:]))
        gt_values[key] = np.array([[(values[j], values[j+1], values[j+2], values[j+3]), values[j+4]] for j in range(0, len(values), 5)])

    pred_values = {}
    for i in y_scores:
        key = i.split(',')[0].strip()
        values = list(map(fn, i.split(',')[1:]))
        pred_values[key] = np.array([[(values[j], values[j+1], values[j+2], values[j+3]), values[j+4]] for j in range(0, len(values), 5)])
        
    assert pred_values.keys() == gt_values.keys(), 'predicted image set and gt image set is not equal'
        
    all_scores = []
    n_gt = 0
    for key in pred_values.keys():
        pred_val, gt_val = pred_values[key], gt_values[key]
        len_pred, len_gt= len(pred_val), len(gt_val)
        
        n_gt += len_gt
        
        b_pairs = {}

        for i in range(len_gt):
            for j in range(len_pred):
                gt_B = gt_val[i]
                pred_B = pred_val[j]
                
                if gt_B[1] == pred_B[1]:
                    b_pairs[f'{i}_{j}'] = compute_IoU(gt_B[0], pred_B[0])
                
        sorted_pairs = sorted(b_pairs.items(), key=lambda kv: kv[1], reverse=True)
        
        IoU_score = get_score(sorted_pairs, len_pred)
        all_scores.append(IoU_score)
                
    click.echo(np.sum(all_scores))
    click.echo(f'n_gt: {n_gt}')
    click.echo(f'n_TP: {n_TP}, n_FP: {n_FP}, n_FN: {n_FN}')

if __name__ == "__main__":
    main()

