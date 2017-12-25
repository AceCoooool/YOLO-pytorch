import torch
import numpy as np
from PIL import ImageFont, ImageDraw


# (x,y,w,h)--->(x1, y1, x2, y2)
def box_to_corners(box_pred):
    box_mins = box_pred[..., 0:2] - (box_pred[..., 2:] / 2.)
    box_maxes = box_pred[..., 0:2] + (box_pred[..., 2:] / 2.)
    return torch.cat([box_mins[..., 0:1], box_mins[..., 1:2],
                      box_maxes[..., 0:1], box_maxes[..., 1:2]], 3)


# remove the proposal detector which is less than the threshold
def filter_box(boxes, box_conf, box_prob, threshold=.5):
    box_scores = box_conf.repeat(1, 1, 1, box_prob.size(3)) * box_prob
    box_class_scores, box_classes = torch.max(box_scores, dim=3)
    prediction_mask = box_class_scores > threshold
    prediction_mask4 = prediction_mask.unsqueeze(3).expand(boxes.size())

    boxes = torch.masked_select(boxes, prediction_mask4).contiguous().view(-1, 4)
    scores = torch.masked_select(box_class_scores, prediction_mask)
    classes = torch.masked_select(box_classes, prediction_mask)
    return boxes, scores, classes


# non-max-suppression process
def non_max_suppression(boxes, scores, overlap=0.5, top_k=200):
    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.is_cuda: keep = keep.cuda()
    if boxes.numel() == 0:
        return keep
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1, yy1, xx2, yy2 = boxes.new(), boxes.new(), boxes.new(), boxes.new()
    w, h = boxes.new(), boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        # check sizes of xx1 and xx2.. after each iteration
        w, h = torch.clamp(xx2 - xx1, min=0.0), torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


# draw proposal boxes
def draw_box(cfg, image, label, box, c):
    w, h = image.size
    font = ImageFont.truetype(font='./config/FiraMono-Medium.otf', size=np.round(3e-2 * h).astype('int32'))
    thickness = (w + h) // 300
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)
    left, top, right, bottom = box
    top, left = max(0, np.round(top).astype('int32')), max(0, np.round(left).astype('int32'))
    right, bottom = min(w, np.round(right).astype('int32')), min(h, np.round(bottom).astype('int32'))
    print(label, (left, top), (right, bottom))
    text_orign = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=cfg.colors[c])
    draw.rectangle([tuple(text_orign), tuple(text_orign + label_size)], fill=cfg.colors[c])
    draw.text(text_orign, label, fill=(0, 0, 0), font=font)
    del draw
