from benchmark.utils import *
import motmetrics as mm
import pandas as pd

def evaluate(pred_path, gt_path, min_vis=0, iou_threshold=0.5,
             metrics=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches']):
    gt_cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
    gt_data = pd.read_csv(gt_path, names=gt_cols)

    pred_cols = ['frame', 'x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright', 'id', 'has_violated']
    pred_data =pd.read_csv(pred_path, names=pred_cols)
    pred_data = convert_pred(pred_data)

    acc = get_mot_accum(pred_data, gt_data, min_vis=min_vis, iou_threshold=iou_threshold)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name='acc')

    return summary