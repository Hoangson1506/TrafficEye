from ultralytics.engine.results import Results
import os
from typing import Optional, List

def inference_video(
        model,
        data_path,
        device: str = 'cpu',
        output_path: Optional[str] = None,
        stream: bool = True,
        conf_threshold = 0.25,
        iou_threshold = 0.5,
        classes: Optional[List[int]] = None,
        **kwargs
) -> Results:
    """Run object detection model and return results

    Args:
        model_path (YOLO): the detection model
        data_path (str): path to input data (image, video, folder, ...)
        output_path (Optional[str], optional): path to output folder for inspection. Defaults to None.
        conf_threshold (float, optional): confidence threshold for box results. Defaults to 0.25.
        iou_threshold (float, optional): IoU threshold for NMS. Defaults to 0.5.

    Returns:
        Results: YOLO results object
    """
    save = output_path is not None
    if save and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    results = model(
        source=data_path,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        project=output_path if save else None,
        stream=stream,
        classes=classes,

        # Visualization options
        save=save,
        save_txt=save,
        save_conf=save,
        **kwargs
    )

    return results