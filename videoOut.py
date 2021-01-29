import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import trange
import numpy as np
from Lenet.model import LeNet


def workcard_rec(video_path, out_path):
    # 获取输入的视频源
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)

    frames_num = int(video.get(7))                        # 获取总帧数
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),     # 获取分辨率
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))                # 获取帧率

    # 定义视频文件输出对象
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter(out_path, fourcc, fps, size)  # 第一个参数是保存视频文件的绝对路径

    # 处理视频并输出
    for i in trange(frames_num):
        _, frame = video.read()
        if frame is None:
            break

        # 处理输入的格式
        frame_tmp = frame[270:340, 290:360]     # 统一采用这部分
        frame_tmp = Image.fromarray(np.uint8(frame_tmp))
        frame_tmp = transform(frame_tmp)  # [C, H, W]
        frame_tmp = torch.unsqueeze(frame_tmp, dim=0)  # [N, C, H, W]
        # 利用LeNet进行预测
        with torch.no_grad():
            outputs = net(frame_tmp)
            predict = torch.max(outputs, dim=1)[1].data.numpy()
        label = classes[int(predict)]
        # 画矩形框
        cv2.rectangle(frame, (290, 270), (360, 340), (0, 255, 0), 4)
        # 标注类别
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (290, 260), font, 0.6, (0, 255, 0), 1)
        # 写输出流
        outVideo.write(frame)

    video.release()
    outVideo.release()


if __name__ == '__main__':
    # 输入视频信息
    video_path = './video/demo2.mp4'
    # 输出视频名称
    out_path = './ouput/output2.mp4'

    # 载入模型和相关参数
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes = ('front', 'others', 'reverse')
    net = LeNet()
    net.load_state_dict(torch.load('./Lenet/Lenet.pth'))

    # 生成新的视频结果
    workcard_rec(video_path, out_path)



