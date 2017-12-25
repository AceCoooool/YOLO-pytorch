import colorsys
import random

# create model information
tiny, voc, num = False, True, 18
flag = [1, 1, 0] * 3 + [1, 0, 1] * 2 + [0, 1, 0]
size_flag = [3, 6, 9, 11, 14, 16]
pool = [0, 1, 4, 7, 13]

# anchor and classes information
anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']
anchor_num = 5
class_num = len(classes)

# color for draw boxes
hsv_tuples = [(x / class_num, 1., 1.) for x in range(class_num)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

# input image information
image_size = (416, 416)
feat_size = image_size[0] // 32

cuda = False

# demo parameter
eval = False
score_threshold = 0.5
nms_threshold = 0.4
iou_threshold = 0.6
