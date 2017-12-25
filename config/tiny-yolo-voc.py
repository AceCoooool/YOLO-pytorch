import colorsys
import random

# create model information
tiny, voc = True, True
num = 7
flag, size_flag = [1] * num, []
pool = [0, 1, 2, 3, 4]

# anchor and classes information
anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
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
