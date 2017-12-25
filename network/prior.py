import torch
import numpy as np
from itertools import product


# prior box in each position
class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg.image_size
        self.num_priors = cfg.anchor_num
        self.feat_size = cfg.feat_size
        self.anchors = np.array(cfg.anchors).reshape(-1, 2)

    def forward(self):
        mean = []
        for i, j in product(range(self.feat_size), repeat=2):
            cx = j
            cy = i
            for k in range(self.num_priors):
                w = self.anchors[k, 0]
                h = self.anchors[k, 1]
                mean += [cx, cy, w, h]

        output = torch.Tensor(mean).view(-1, 4)
        return output
