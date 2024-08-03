import numpy as np

def calculate_metrics(mask1, mask2):
    """ Calculate IoU and F1 score for two masks """
    if mask1.shape != mask2.shape:
        raise ValueError("Both masks should have the same shape")

    mask1 = np.asarray(mask1).astype(np.bool_)
    mask2 = np.asarray(mask2).astype(np.bool_)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    f1 = (2. * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2))

    return iou, f1