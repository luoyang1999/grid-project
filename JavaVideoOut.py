import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import trange
import numpy as np
from Lenet.model import LeNet
import argparse
import pymysql
import shutil


# 定义全局变量
n = 0   # 定义鼠标按下的次数
x1 = 0  # x,y 坐标的临时存储
y1 = 0
x2 = 0
y2 = 0


def getXY(hall_name):
    global x1, y1, x2, y2
    db = pymysql.connect("localhost", "root", "123456", "surveillance", charset="utf8")
    cursor = db.cursor()
    sql = "select * from t_anchor where hall_name=" + '"' + hall_name + '"'
    print(sql)
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        result = cursor.fetchone()
        # 获取所有记录列表
        x1 = result[1]
        y1 = result[2]
        x2 = result[3]
        y2 = result[4]
    except:
        print("Error: unable to fetch data")
    db.close()


def workcard_rec(video_path, out_path):
    # 获取输入的视频源
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)

    frames_num = int(video.get(7))  # 获取总帧数
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),  # 获取分辨率
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))  # 获取帧率

    # 定义视频文件输出对象
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

    outVideo = cv2.VideoWriter(out_path + "/wordcard.mp4", fourcc, fps, size)  # 第一个参数是保存视频文件的绝对路径

    # 定义输出的记录结果
    f = open(out_path + "/wordcard.txt", "w")

    # 视频窗口名称
    cv2.namedWindow('window')

    # 处理视频并输出
    for i in trange(frames_num):
        # 获取一帧图像
        _, frame = video.read()
        if frame is None:
            break

        # 处理输入的格式
        frame_tmp = frame[y1:y2, x1:x2]  # 统一采用这部分[270:340, 290:360]
        frame_tmp = Image.fromarray(np.uint8(frame_tmp))
        frame_tmp = transform(frame_tmp)  # [C, H, W]
        frame_tmp = torch.unsqueeze(frame_tmp, dim=0)  # [N, C, H, W]
        # 利用LeNet进行预测
        with torch.no_grad():
            outputs = net(frame_tmp)
            predict = torch.max(outputs, dim=1)[1].data.numpy()
        label = classes[int(predict)]

        # 画矩形框(290, 270), (360, 340) x1 y1 x2 y2
        # print(x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        # 标注类别
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (x1, y1-10), font, 0.6, (0, 255, 0), 1)
        # 写入txt文件保存
        f.write(str(int(i/fps)) + " " + label + "\n")
        # 写输出流
        outVideo.write(frame)
        # 显示视频
        cv2.imshow("window", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            f.close()
            break

    f.close()
    video.release()
    outVideo.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./video/吕家坨.mp4", help="输入视频地址")
    parser.add_argument("-o", "--output", default="./output/demo1.mp4", help="输出视频地址")
    parser.add_argument("-n", "--name", default="吕家坨", help="营业厅名称")
    args = parser.parse_args()

    # 输入视频信息
    video_path = args.input
    # 输出视频名称
    out_path = args.output

    # 载入模型和相关参数
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes = ('front', 'reverse')
    net = LeNet()
    net.load_state_dict(torch.load('./Lenet/Lenet.pth'))

    # 获取视频建材范围
    getXY(args.name)

    # 将视频输出到无英文路径
    tmp_path = "C:/Users/13216/IdeaProjects/ssm_springmvc/surveillance/target/classes/static/files/tmp"

    # 生成新的视频结果
    workcard_rec(video_path, tmp_path)

    # 对生成的视频进行移动
    shutil.move(tmp_path, out_path)
    # 对生成的txt文本进行移动
    shutil.move()
