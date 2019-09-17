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

stats = {
    0: { # yaya 
        'total': 0,
        'n_TP': 0,
        'n_FP': 0,
        'n_FN': 0
    },
    1: { # arac
        'total': 0,
        'n_TP': 0,
        'n_FP': 0,
        'n_FN': 0
    }
}

precision = lambda TP, FP: TP / (TP + FP) 
recall = lambda TP, FN: TP / (TP + FN) 

def get_score(y_class, y_pred, pred_ind, IoU_scores, threshold):
    global stats;
    scores = []
    temp = []
    
    inds = np.argsort(-IoU_scores)
    
    for i, j in zip(inds, pred_ind[inds]):
        y_c = y_class[j]
        y_p = y_pred[i]
        if y_c != y_p:
            stats[y_c]['n_FP'] += 1
            scores.append(-1.0)
            continue
        IoU_score = IoU_scores[i]
        if IoU_score >= threshold:
            if pred_ind[i] not in temp:
                stats[y_c]['n_TP'] += 1
                scores.append(3 * IoU_score)
        else:
            stats[y_c]['n_FP'] += 1
            scores.append(-1 * (1 - IoU_score))
            
        temp.append(pred_ind[i])
    
    # Add not found GT boxes
    scores += [-1.0] * (len(y_class) - len(np.unique(pred_ind)))
    stats[y_c]['n_FN'] += (len(y_class) - len(np.unique(pred_ind)))
        
    return np.sum(scores)

def get_metric_IoU(y_true, y_pred, threshold=0.6):
    all_scores = []
    for y_t, y_p in zip(y_true, y_pred):
        y_t_bboxes = y_t[0]
        y_p_bboxes = y_p[0]

        y_t_labels = y_t[1]
        y_p_labels = y_p[1]

        len_pred, len_gt = len(y_p_bboxes), len(y_t_bboxes)
                
        if not len(y_t_bboxes):
            all_scores.append(-1 * len_pred)
            continue
        
        u_c, cnt = np.unique(y_t_labels, return_counts=True)
        for cl, ct in zip(u_c, cnt):
            stats[cl]['total'] += ct
        
        score_arr = np.zeros((len_pred, len_gt))
        for i in range(len_pred):
            for j in range(len_gt):
                score_arr[i, j] = compute_IoU(y_p_bboxes[i], y_t_bboxes[j])
                
        
        if len_pred:
            scores = score_arr[np.arange(len(score_arr)), score_arr.argmax(axis=1)]

            IoU_score = get_score(y_t_labels, y_p_labels, score_arr.argmax(axis=1), scores, threshold)
        else:
            IoU_score = -1 * (len_gt - len_pred)
        all_scores.append(IoU_score)

    return np.sum(all_scores)

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
        
    y_true = []
    y_pred = []
    for key in pred_values.keys():
        pred_val, gt_val = pred_values[key], gt_values[key]

        gt_bboxes = gt_val[:, 0]
        pred_bboxes = pred_val[:, 0]

        gt_labels = gt_val[:, 1]
        pred_labels = pred_val[:, 1]
        
        y_true.append((gt_bboxes, gt_labels))
        y_pred.append((pred_bboxes, pred_labels))

    all_score = get_metric_IoU(y_true, y_pred, threshold)
    
    print(stats)
    click.echo(f'Total Box: {stats[0]["total"] + stats[1]["total"]}')
    click.echo('\n================ T3 METRIC ================')
    click.echo(f'{all_score:.3f} over {3 * (stats[0]["total"] + stats[1]["total"])}')
    click.echo('\n=========== PRECISION & RECALL ============')
    for k, v in stats.items():
        cl = 'yaya' if k == 0 else 'arac'
        if stats[k]['total'] == 0:
            continue
            
        pre = precision(stats[k]['n_TP'], stats[k]['n_FP'])
        rec = recall(stats[k]['n_TP'], stats[k]['n_FN'])
        click.echo(f"{cl}: \n\tprecision: {pre:.3f} \n\trecall: {rec:.3f}\n\tavg: {(pre + rec) / 2:.3f}")
        
#     click.echo(f'n_gt: {n_gt}')
#     click.echo(f'n_TP: {n_TP}, n_FP: {n_FP}, n_FN: {n_FN}')

if __name__ == "__main__":
    main()
