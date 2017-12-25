from torch.autograd import Function
from network.postprocess import box_to_corners, filter_box, non_max_suppression


# test phase's proposal process
class Detect(Function):
    def __init__(self, cfg, eval=False):
        self.class_num = cfg.class_num
        self.feat_size = cfg.feat_size
        if eval:
            self.nms_t, self.score_t = cfg.eval_nms_threshold, cfg.eval_score_threshold
        else:
            self.nms_t, self.score_t = cfg.nms_threshold, cfg.score_threshold

    def forward(self, box_pred, box_conf, box_prob, priors, img_shape, max_boxes=10):
        box_pred[..., 0:2] += priors[..., 0:2]
        box_pred[..., 2:] *= priors[..., 2:]
        boxes = box_to_corners(box_pred) / self.feat_size
        boxes, scores, classes = filter_box(boxes, box_conf, box_prob, self.score_t)
        if boxes.numel() == 0:
            return boxes, scores, classes
        boxes = boxes * img_shape.repeat(boxes.size(0), 1)
        keep, count = non_max_suppression(boxes, scores, self.nms_t)
        boxes = boxes[keep[:count]]
        scores = scores[keep[:count]]
        classes = classes[keep[:count]]
        return boxes, scores, classes
