import os
import cv2
import torch
import argparse
import importlib
from torch.autograd import Variable
from network.postprocesscv import draw_box_cv
from network.yolo import yolo


def demo_cam(cfg, save_path, save):
    net = yolo(cfg)
    net.load_state_dict(torch.load(cfg.trained_model))
    if cfg.cuda: net = net.cuda()
    net.eval()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened(): raise IOError("check your camera or the opencv library...")
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path + '/out_camera.avi', fourcc, 20.0, (640, 480))
    while cam.isOpened():
        ret, image = cam.read()
        if ret:
            reimg = cv2.resize(image, cfg.image_size)
            reimg = torch.from_numpy(reimg.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            reimg = Variable(reimg, volatile=True)
            if cfg.cuda: reimg = reimg.cuda()
            boxes, scores, classes = net(reimg, (image.shape[1], image.shape[0]))
            boxes, scores, classes = boxes.data.cpu(), scores.data.cpu(), classes.data.cpu()
            for i, c in list(enumerate(classes)):
                pred_class, box, score = cfg.classes[c], boxes[i], scores[i]
                label = '{} {:.2f}'.format(pred_class, score)
                draw_box_cv(cfg, image, label, box, c)
            cv2.imshow('camera', image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if save: out.write(image)
    cam.release()
    if save: out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # default path
    curdir = os.getcwd()
    trained_model = os.path.join(curdir, 'model/yolo.pth')

    # demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_type', default='yolo', type=str)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--save_path', default='results/demo/camera', type=str)
    parser.add_argument('--nms_threshold', default=0.4, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--trained_model', default=trained_model, type=str)
    parser.add_argument('--cuda', default=True, type=bool)

    config = parser.parse_args()
    cfg = importlib.import_module('config.' + config.yolo_type)
    cfg.nms_threshold = config.nms_threshold
    cfg.score_threshold = config.score_threshold
    cfg.trained_model = config.trained_model
    cfg.cuda = config.cuda

    if not os.path.exists(config.save_path) and config.save: os.mkdir(config.save_path)
    demo_cam(cfg, config.save_path, config.save)
