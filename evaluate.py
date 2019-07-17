import os
import click
from tabulate import tabulate
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

def get_score(y_class, y_pred, pred_ind, IoU_scores, threshold):
    scores = []
    temp = []
    
    for i, j in enumerate(pred_ind):
        y_c = y_class[j]
        y_p = y_pred[i]
        if y_c != y_p:
            scores.append(-1.0)
            continue
        IoU_score = IoU_scores[i]
        if IoU_score >= threshold:
            if pred_ind[i] not in temp:
                scores.append(3 * IoU_score)
        else:
            scores.append(-1 * (1 - IoU_score))
            
        temp.append(pred_ind[i])
        
    return scores

fn = lambda x: int(x.strip())

@click.command()
@click.argument('y_true', type=click.File('r'))
@click.argument('y_scores', type=click.File('r'))
@click.option('--threshold', type=click.FloatRange(0, 1), default=0.6)
def main(y_true, y_scores, threshold):
    gt_values = {}
    for i in y_true:
        key = i.split(',')[0]
        values = list(map(fn, i.split(',')[1:]))
        gt_values[key] = np.array([[(values[j], values[j+1], values[j+2], values[j+3]), values[j+4]] for j in range(0, len(values), 5)])

    pred_values = {}
    for i in y_scores:
        key = i.split(',')[0]
        values = list(map(fn, i.split(',')[1:]))
        pred_values[key] = np.array([[(values[j], values[j+1], values[j+2], values[j+3]), values[j+4]] for j in range(0, len(values), 5)])
        
    assert pred_values.keys() == gt_values.keys(), 'predicted image set and gt image set is not equal'
        
    all_scores = []
    for key in pred_values.keys():
        pred_val, gt_val = pred_values[key], gt_values[key]
        len_pred, len_gt= len(pred_val), len(gt_val)
        score_arr = np.zeros((len_pred, len_gt))
        for i in range(len_pred):
            for j in range(len_gt):
                score_arr[i, j] = compute_IoU(pred_val[i][0], gt_val[j][0])
                
        y_class = gt_val[:, 1]
        y_pred = pred_val[:, 1]
        scores = score_arr[np.arange(len(score_arr)), score_arr.argmax(axis=1)]
        
        IoU_score = get_score(y_class, y_pred, score_arr.argmax(axis=1), scores, threshold)
        all_scores.append(IoU_score)
                
    click.echo(all_scores)

if __name__ == "__main__":
    main()