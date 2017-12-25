import os
import torch
import argparse
import importlib
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from network.postprocess import draw_box
from network.yolo import yolo


# TODO: add a dataloader form
def demo(cfg, img_list, save_path):
    transform = transforms.Compose([transforms.Resize(cfg.image_size), transforms.ToTensor()])
    net = yolo(cfg)
    net.load_state_dict(torch.load(cfg.trained_model))
    if cfg.cuda: net = net.cuda()
    net.eval()
    for img in img_list:
        image = Image.open(img)
        reimg = Variable(transform(image).unsqueeze(0))
        if cfg.cuda: reimg = reimg.cuda()
        boxes, scores, classes = net(reimg, image.size)
        boxes, scores, classes = boxes.data.cpu(), scores.data.cpu(), classes.data.cpu()
        print('Find {} boxes for {}.'.format(len(boxes), img.split('/')[-1]))
        for i, c in list(enumerate(classes)):
            pred_class, box, score = cfg.classes[c], boxes[i], scores[i]
            label = '{} {:.2f}'.format(pred_class, score)
            draw_box(cfg, image, label, box, c)
        image.save(os.path.join(save_path, img.split('/')[-1]), quality=90)


if __name__ == '__main__':
    # default path
    curdir = os.getcwd()
    demo_path = os.path.join(curdir, 'results/demo')
    trained_model = os.path.join(curdir, 'model/yolo.pth')

    # demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', default=demo_path, type=str)
    parser.add_argument('--yolo_type', default='yolo', type=str)
    parser.add_argument('--nms_threshold', default=0.4, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--trained_model', default=trained_model, type=str)
    parser.add_argument('--cuda', default=True, type=str)

    config = parser.parse_args()
    cfg = importlib.import_module('config.' + config.yolo_type)
    cfg.nms_threshold = config.nms_threshold
    cfg.score_threshold = config.score_threshold
    cfg.demo_path = config.demo_path
    cfg.trained_model = config.trained_model
    cfg.cuda = config.cuda

    ext = ['.jpg', '.png']
    if os.path.isfile(cfg.demo_path) and os.path.splitext(cfg.demo_path)[-1] in ext:
        img_list = [cfg.demo_path]
        save_path = os.path.join(os.path.dirname(cfg.demo_path), 'out')
    elif os.path.isdir(cfg.demo_path):
        imgs = [fname for fname in os.listdir(cfg.demo_path) if os.path.splitext(fname)[-1] in ext]
        img_list = [os.path.join(cfg.demo_path, fname) for fname in imgs]
        save_path = os.path.join(cfg.demo_path, 'out')
    if not os.path.exists(save_path): os.mkdir(save_path)
    if not img_list:
        raise IOError("illegal demo path ...")
    demo(cfg, img_list, save_path)
