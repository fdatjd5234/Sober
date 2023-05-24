import argparse
import threading
import time
from pathlib import Path

import pymysql
import snap7

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# 预测函数
def detect(save_img=False, num_images=1):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # 保存推理结果的图像
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 自定义结果输出
    list_result = []  # 存储检测结果的列表
    path_img = ''  # 图像路径
    dict_original = {'count': 0, 'result': '', 'str': ''}  # 检测结果的原始字典
    dict_result = dict_original.copy()  # 创建一个副本

    # 目录设置
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 递增运行目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # 仅支持CUDA的半精度

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步长
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸
    if half:
        model.half()  # 转换为FP16

    # 第二阶段分类器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化分类器
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 设置为True以加快图像尺寸推断速度
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取类别名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 运行推断
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 运行一次推理以初始化模型
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # 将图像从uint8转换为fp16/32
        img /= 255.0  # 将图像像素值从0-255缩放到0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 应用非极大值抑制
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 应用分类器
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图像的检测结果
            if webcam:  # 批处理大小大于等于1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # 转换为路径对象
            save_path = str(save_dir / p.name)  # 图像保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 标签保存路径
            s += '%gx%g ' % img.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
            if len(det):
                # 将边界框从img_size缩放到im0的尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                dict_result = dict_original.copy()  # 创建一个副本
                str_temp = ''
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串中
                    dict_result[names[int(c)]] = int(n)
                    str_temp += f"{n} {names[int(c)]}{'str_temp' * (n > 1)}, "

                dict_result['count'] = len(det[:, -1].unique())
                dict_result['result'] = str_temp
                dict_result['str'] = s
                list_result.append(dict_result)

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 写入到文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # 标签格式
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # 在图像上绘制边界框
                        color_dict = {"1": [0, 128, 0], "2": [0, 0, 255]}
                        if names[int(cls)] == "good_wheel":
                            color_single = color_dict["1"]
                        else:
                            color_single = color_dict["2"]
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=color_single, line_thickness=3)

            # 打印时间（推理+NMS）
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 流式输出结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1毫秒

            # 保存结果（带有检测结果的图像）
            if save_img:
                for i in range(num_images):
                    if dataset.mode == 'image':
                        image_save_path = save_path.replace('.jpg', f'_{i+1}.jpg')
                        cv2.imwrite(image_save_path, im0)
                    else:  # 视频或流式数据
                        if vid_path != save_path:  # 新视频
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # 释放之前的视频编写器
                            if vid_cap:  # 视频
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # 流式数据
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(list_result)
    # print(list_result[0].get('bad_wheel'))
    return list_result  # 返回检测结果列表


""" 原代码---曲洋
# 相机调用
def camera():
    url = "rtsp://admin:cp6_ca123@10.200.35.250/Streaming/Channels/2"
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cv2.imwrite(r"..\images\pictures.jpg", frame)
    print("save pictures.jpg successfully!")
    print("-----------------------------")
    detect()
    cap.release()
"""


def camera(num_images=1):
    # 摄像头URL列表
    urls = [
        "rtsp://admin:password1@camera1_ip/Streaming/Channels/2",  # 第一个摄像头的URL
        "rtsp://admin:password2@camera2_ip/Streaming/Channels/2",  # 第二个摄像头的URL
        "rtsp://admin:password3@camera3_ip/Streaming/Channels/2",  # 第三个摄像头的URL
        "rtsp://admin:password4@camera4_ip/Streaming/Channels/2",  # 第四个摄像头的URL
    ]
    # 保存路径
    save_path = r"..\images\pictures"
    """
    save_path = r"D:\my_images"
    """
    # 创建摄像头对象列表
    caps = [cv2.VideoCapture(url) for url in urls]
    # 循环读取摄像头图像并进行处理
    # 逐个读取摄像头图像并保存
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            file_name = f"{save_path}_{i+1}.jpg"  # 构造文件名，例如：pictures_1.jpg, pictures_2.jpg, ...
            cv2.imwrite(file_name, frame)
            print(f"Saved image: {file_name}")
            # 调用detect函数对照片进行检测
            detect(num_images=num_images)
        else:
            print(f"Failed to read image from camera")
        # 1秒的等待时间，即每秒处理一次摄像头图像
        time.sleep(1)
    # 释放摄像头对象
    for cap in caps:
        cap.release()


# PLC通讯 弃用
class ConnPlc:
    def __init__(self):
        self.ip = "10.200.35.3"
        self.rock = 0
        self.slot = 2

    def connect(self):
        plc = snap7.client.Client()
        plc.connect(self.ip, self.rock, self.slot)
        return plc

    def getByMb(self):
        plc = self.connect()
        data = plc.mb_read(668, 1)
        data = snap7.util.get_bool(data, 0, 0)
        return data


# mysql数据库通讯
class ConMysql:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 3306
        self.user = "root"
        self.pwd = "123456"
        self.database = "plc"
        self.charset = "utf8"

    def conn(self):
        conn = pymysql.connect(host=self.host, user=self.user, password=self.pwd, database=self.database, port=self.port
                               , charset=self.charset)
        return conn

    def readMysql(self, con):
        cur = con.cursor()
        sql = "SELECT * FROM `plcsingle`"
        # sql = "SELECT plcsingle.inPosition FROM `plcsingle`"
        cur.execute(sql)
        res = cur.fetchone()
        cur.close()
        con.close()
        return res[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp21/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'..\images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=r'..\detects', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True,
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    mysql = ConMysql()

    lastValue = 0
    while True:
        con = mysql.conn()
        res = mysql.readMysql(con)
        if res == 0 or res == lastValue:
            pass
        else:
            thread = threading.Thread(target=camera, name='camera')
            thread.start()
            print(res)

        lastValue = res
        time.sleep(2)
