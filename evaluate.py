from benchmark.metrics import evaluate
from utils.parse_args import parse_args_eval
import numpy as np


if __name__ == "__main__":
    args = parse_args_eval()

    pred_path = args.pred_path
    gt_path = args.gt_path
    min_vis = args.min_vis
    iou_threshold = args.iou_threshold
    metrics = args.metrics
    np.random.seed(42)

    summary = evaluate(pred_path=pred_path, gt_path=gt_path, 
                       min_vis=min_vis, iou_threshold=iou_threshold,
                       metrics=metrics)
    
    print(summary)