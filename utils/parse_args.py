import argparse

def parse_args_tracking():
    parser = argparse.ArgumentParser(description="Object tracking script")
    parser.add_argument(
        '--data_path', 
        type=str, 
        default="data/traffic_video.avi",
        help='Path to the input video file (e.g., data/video.mp4).'
    )
    parser.add_argument(
        "--vehicle_model",
        type=str,
        default="detect_gtvn.pt",
        help="Path to vehicle detection model weights."
    )
    parser.add_argument(
        "--license_model",
        type=str,
        default="lp_yolo11s.pt",
        help="Path to license detection model weights."
    )
    parser.add_argument(
        "--character_model",
        type=str,
        default="yolo11s.pt",
        help="Path to character detection model weights."
    )
    parser.add_argument(
        '--tracker', 
        type=str, 
        default='sort', 
        choices=['sort', 'bytetrack'],
        help='The tracking algorithm to use: sort or bytetrack.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output', 
        help='Directory to save the resulting video.'
    )
    parser.add_argument(
        "--save",
        type=str,
        default='True',
        help='Save video if set to True, else do not save'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        help='Used device to run tracking.'
    )
    parser.add_argument(
        '--light_detect',
        type=str,
        default='True',
        choices=['True', 'False'],
        help='Enable traffic light detection.'
    )
    args = parser.parse_args()
    return args

def parse_args_eval():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        '--pred_path', 
        type=str, 
        default="output/csv/MOT16-02-raw_bytetrack.csv",
        help='Path to the tracking output csv file (e.g., output/csv/Name.csv).'
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="data/train/MOT16-02/gt/gt.txt",
        help="Path to ground truth text file."
    )
    parser.add_argument(
        '--min_vis', 
        type=float, 
        default=0, 
        help='min visibility to filter ground truth with low visibilities (hard to detect).'
    )
    parser.add_argument(
        '--iou_threshold', 
        type=float, 
        default=0.5, 
        help='The IoU threshold for evaluation.'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 
                 'num_false_positives', 'num_misses', 'num_switches'],
        choices=['num_frames', 'num_matches', 'num_switches', 'num_false_positives', 
                 'num_misses', 'num_detections', 'num_objects', 'num_predictions',
                 'num_unique_objects', 'mostly_tracked', 'partially_tracked',
                 'mostly_lost', 'num_fragmentations', 'motp', 'mota', 'precision',
                 'recall', 'idfp', 'idfn', 'idtp', 'idp', 'idr', 'idf1', 'obj_frequencies',
                 'pred_frequencies', 'track_ratios', 'id_global_assignment', 'deta_alpha',
                 'assa_alpha', 'hota_alpha'],
        help='metrics needed to compute. details at https://github.com/cheind/py-motmetrics'
    )
    args = parser.parse_args()
    return args