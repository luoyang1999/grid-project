import os
import cv2

# 定义全局变量
n = 0  # 定义鼠标按下的次数
x1 = 0 # x,y 坐标的临时存储
y1 = 0
x2 = 0
y2 = 0

# 鼠标点击事件
def draw_rectangle(event, x, y, flags, param):
    global n, x1, y1, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        if n == 0:  # 首次按下保存坐标值
            n += 1
            x1, y1 = x, y
            cv2.circle(param, (x1, y1), 2, (255, 255, 255), -1)  # 第一次打点
        else:  # 第二次按下显示矩形
            n = 0
            x2, y2 = x, y
            cv2.rectangle(param, (x1, y1), (x2, y2), (255, 255, 255), 3)  # 第二次画矩形


def extract_frames(video_path, video_name, dst_folder, index, frame_num):

    video = cv2.VideoCapture()
    if not video.open(video_path + "/" + video_name):
        print("can not open the video")
        exit(1)
    count = 0
    frames_num = video.get(7)   # 5000
    step = int(frames_num/frame_num) #if you want to gap

    # 视频窗口名称
    cv2.namedWindow('window')

    # step = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break

        # 第一帧鼠标标定识别区域
        if count == 0:
            cv2.setMouseCallback('window', draw_rectangle, frame)
            while True:
                cv2.imshow('window', frame)
                if cv2.waitKey(20) & 0xFF == 13:
                    break

        if count % step == 0:
            save_path = "{}/{}/{} {:>04d}.png".format(dst_folder, video_name.split(".")[0], video_name.split(".")[0], index)
            print(save_path)
            # [325:360, 330:400]
            frame = frame[y1:y2, x1:x2]                    # 设置抽帧区域270:340, 290:360
            # cv2.imwrite(save_path, frame)
            cv2.imencode('.png', frame)[-1].tofile(save_path)  # 正确方法
            index += 1
        count += 1

        if index == frames_num:
            break

    video.release()


if __name__ == '__main__':
    video_path = 'F:\冀北视频\汤家河'
    video_list = os.listdir(video_path)
    print(video_list)

    ims_folder = './frames'
    if not os.path.exists(ims_folder):
        os.makedirs(ims_folder)
    index = 0
    target_num = 300

    # for video_name in video_list:
    #     filepath = ims_folder + "/" + video_name.split(".")[0]
    #     if not os.path.exists(filepath):
    #         os.makedirs(filepath)
    #     extract_frames(video_path, video_name, ims_folder, index, target_num)


    extract_frames(video_path, "汤家河 11月4日 13时20分至13时40分.mp4", ims_folder, index, target_num)