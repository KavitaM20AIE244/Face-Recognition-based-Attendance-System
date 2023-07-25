import numpy as np

def calculate_iou(box1, box2):

    """Calculate the Intersection over Union (IoU) between two bounding boxes
    Each box is list in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]"""
    
    x1, y1, w1, h1 = box1[0],box1[1],box1[3],box1[4]
    x2, y2, w2, h2 = box2[0],box2[1],box2[3],box2[4]
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou


def nms(detections, threshold=0.5):
    """
    This function performs Non-Maxima Suppression (NMS) on a list of detections.

    Args:
        detections (list): A list of detections, where each detection is represented as [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection].
        threshold (float): The threshold used for NMS. If the IoU of two detections is greater than this threshold, the detection with the lower confidence score will be suppressed.

    Returns:
        list: A list of detections after NMS.

    The function first sorts the detections in descending order of their confidence score.
    It then initializes a new list to store the final detections and appends the detection with the highest confidence score to this list.
    For each detection in the remaining list, the function calculates the intersection over union (IoU) with all the detections in the new list.
    If the IoU of the current detection with any of the detections in the new list is greater than the threshold, the current detection is suppressed.
    Otherwise, it is added to the new list.
    The function returns the new list of detections after NMS.
    """
    if len(detections) == 0:
        return []

    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda x: x[2], reverse=True)

    # Initialize a list to store the final detections
    new_detections = []

    # Append the detection with the highest confidence score to the new list
    new_detections.append(detections[0])

    # Remove the detection with the highest confidence score from the original list
    del detections[0]

    # For each detection in the remaining list, calculate the IoU with all the detections in the new list
    # If the IoU is greater than the threshold, suppress the detection. Otherwise, add it to the new list.
    for detection in detections:
        overlap = False
        for new_detection in new_detections:
            iou = calculate_iou(detection, new_detection)
            if iou > threshold:
                overlap = True
                break
        if not overlap:
            new_detections.append(detection)

    return new_detections
