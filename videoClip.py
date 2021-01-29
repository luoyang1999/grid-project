import os
import cv2


def extract_frames(video_path, video_name, dst_folder, index, frame_num):

    video = cv2.VideoCapture()
    if not video.open(video_path + "/" + video_name):
        print("can not open the video")
        exit(1)
    count = 0
    frames_num = video.get(7)   # 5000
    step = int(frames_num/frame_num) #if you want to gap

    # step = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % step == 0:
            save_path = "{}/{}/{} {:>04d}.png".format(dst_folder, video_name.split(".")[0], video_name.split(".")[0], index)
            print(save_path)
            # [325:360, 330:400]
            frame = frame[270:340, 290:360]                    # 设置抽帧区域
            # cv2.imwrite(save_path, frame)
            cv2.imencode('.png', frame)[-1].tofile(save_path)  # 正确方法
            index += 1
        count += 1

        if index == frames_num:
            break

    video.release()


if __name__ == '__main__':
    video_path = 'F:\冀北视频\吕家坨1'
    video_list = os.listdir(video_path)
    print(video_list)

    ims_folder = './frames'
    if not os.path.exists(ims_folder):
        os.makedirs(ims_folder)
    index = 0
    target_num = 100

    for video_name in video_list:
        filepath = ims_folder + "/" + video_name.split(".")[0]
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        extract_frames(video_path, video_name, ims_folder, index, target_num)


    # extract_frames(video_path, "吕家坨 11月3日 16时15分至16时40分.mp4", ims_folder, index, target_num)