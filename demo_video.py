import os
import cv2
import torch
import argparse
import importlib
from torch.autograd import Variable
from timeit import default_timer as timer
from network.postprocesscv import draw_box_cv
from network.yolo import yolo


def demo_video(cfg, file, save_path, save, start_frame=0):
    net = yolo(cfg)
    net.load_state_dict(torch.load(cfg.trained_model))
    if cfg.cuda: net = net.cuda()
    net.eval()
    video = cv2.VideoCapture(file)
    if not video.isOpened():
        raise IOError('Could not open video, check your opencv and video')
    if save:
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(os.path.join(save_path + file.split('/')[-1]), cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    if start_frame > 0:
        video.set(cv2.CAP_PROP_POS_MSEC, start_frame)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: "
    prev_time = timer()

    while True:
        retval, orig_image = video.read()
        if not retval:
            print('Done !!!')
            return
        reimg = cv2.resize(orig_image, cfg.image_size)
        reimg = torch.from_numpy(reimg.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        x = Variable(reimg, volatile=True)
        if cfg.cuda: x = x.cuda()
        boxes, scores, classes = net(x, (orig_image.shape[1], orig_image.shape[0]))
        boxes, scores, classes = boxes.data.cpu(), scores.data.cpu(), classes.data.cpu()
        for i, c in list(enumerate(classes)):
            pred_class, box, score = cfg.classes[c], boxes[i], scores[i]
            label = '{} {:.2f}'.format(pred_class, score)
            draw_box_cv(cfg, orig_image, label, box, c)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time += exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time -= 1
            fps = "FPS:" + str(curr_fps)
            curr_fps = 0
        cv2.rectangle(orig_image, (0, 0), (50, 17), (255, 255, 255), -1)
        cv2.putText(orig_image, fps, (3, 10), 0, 0.35, (0, 0, 0), 1)
        if save:
            out.write(orig_image)
        cv2.imshow("yolo result", orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    if save:
        out.release()


if __name__ == '__main__':
    # default path
    curdir = os.getcwd()
    trained_model = os.path.join(curdir, 'model/yolo.pth')
    demo_path = os.path.join(curdir, 'results/demo/video.avi')

    # demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_type', default='yolo', type=str)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--save_path', default='results/demo/video', type=str)
    parser.add_argument('--demo_path', default=demo_path, type=str)
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

    ext = ['.mp4', '.avi']
    if not os.path.splitext(config.demo_path)[-1] in ext:
        raise IOError("illegal video form...")
    if not os.path.exists(config.save_path) and config.save: os.mkdir(config.save_path)
    demo_video(cfg, config.demo_path, config.save_path, config.save)
