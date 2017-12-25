# YOLO-Pytorch

[English](README.md)

## 说明

主要将Keras版本的Yolo2[YAD2K](https://github.com/allanzelener/YAD2K)移植到pytorch。

原始论文：[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

<p align="center"><img width="60%" src="png/cat.jpg" /></p>

---

## 环境要求

- Pytorch 0.3.0
- torchvision
- OpenCV
- python 3

## 使用说明

1. 从[官网](http://pjreddie.com/darknet/yolo/)下载已经训练好的模型和说明：

   ```bash
   # 还有tiny-yolo-voc 和 yolo-voc等版本
   wget http://pjreddie.com/media/files/yolo.weights
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
   ```

2. 将`.weight`的参数导成`.pth`文件（具体实现见`tools/yad2t.py`函数)：

   ```bash
   python tools/yad2t.py path-to-yolo-cfg path-to-yolo-weights path-to-output-folder
   ```

   说明：① 默认情况假设你将1中下载的参数和说明放置在`config`这个文件夹中   ② 默认会将导出的`.pth`文件保存到`model`这个文件夹中 

3. 三个示例程序：

   - `demo.py`：处理单张图片或者一个包含图片的文件夹

     ```bash
     python demo.py pic-path yolo-type --cuda=True
     ```

     说明： ① 默认图片地址为`results/demo`中的所有图片  ② 默认采用`yolo`这个模型，你可以选择`yolo-voc`或者`tiny-yolo-voc`（但需同前面两步获得对应的训练好的模型）  ③ 是否选择采用gpu模型

   - `demo_cam.py`：摄像头处理（最好是采用gpu加速）

     ```bash
     python demo_cam.py --trained_model=pth_model_from_1
     ```

     说明：① 具体其他参数等请看`demo_cam.py`

   - `demo_video.py`：视频处理（可能有bug）

     ```bash
     python demo_video.py --demo_path=video_path --trained_model=pth_model_from_1
     ```

     说明：① 目前可以处理`avi`和`mp4`格式，其他类型未验证

### 各文件夹说明

- `config`：预先保存的一些参数
- `network`：包含网络结构的实现，图像后处理等操作
- `tools`：包含参数转换等工具
- `results`：主要保存实验结果

## 待办事项

- [ ] 测试mAP
- [ ] 增加训练过程(这部分待定)

## 问题

欢迎指出存在的bug，以及pull request~ 谢谢