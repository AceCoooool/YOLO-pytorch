import numpy as np
import cv2


# draw proposal boxes using OpenCV
def draw_box_cv(cfg, image, label, box, c):
    h, w = image.shape[:2]
    thickness = (w + h) // 300
    left, top, right, bottom = box
    top, left = max(0, np.round(top).astype('int32')), max(0, np.round(left).astype('int32'))
    right, bottom = min(w, np.round(right).astype('int32')), min(h, np.round(bottom).astype('int32'))
    cv2.rectangle(image, (left, top), (right, bottom), cfg.colors[c], thickness)
    cv2.putText(image, label, (left, top - 5), 0, 0.5, cfg.colors[c], 1)
