# 工牌正反识别

L.Y.	2021.1.28

### 1 使用

##### 1.1 获取固定区域画面

​	运行videoClip.py 抽帧区域为 [270:340, 290:360]

```python
video_path = 'F:\冀北视频\吕家坨1'		# 修改视频文件夹
video_list = os.listdir(video_path)
print(video_list)

ims_folder = './frames'				#  修改输出文件夹
if not os.path.exists(ims_folder):
    os.makedirs(ims_folder)
index = 0
target_num = 100					#  设定抽帧数量

frame = frame[270:340, 290:360]     # 设置抽帧区域
```

##### 1.2 划分数据集

​	运行dateSplit.py 划分比例 train:test:val = 0.8:0.1:0.1

```python
src_data_folder = "./frames/images"
target_data_folder = "./frames/target"
```

##### 1.3 训练

​	运行train.py,设置epoch=30 lr=0.002 batch_size=36 保存模型权重

```python
save_path = Lenet1.pth
torch.save(net.state_dict(), save_path)
```

##### 1.4 预测

​	运行videoOut.py修改输入输出文件路径，运行时鼠标左键点击两下，Enter确定识别区域，
    即可获得输出文件，中途按下ESC可停止
    
```python
# 输入视频信息
video_path = './video/demo2.mp4'
# 输出视频名称
out_path = './output/output2.mp4'
```

### 2 环境

- python 3.7

- torchvision	0.5.0

- torch 1.4.0


### 3 目录

```
###########目录结构描述
├── Readme.md                   // help
├── videoOut.py                 // 识别视频中的工牌
├── videoClip.py                // 提取视频中工牌存为png
├── dataSplit.py                // 划分数据集为tarin/test/val
├── Lenet                       // 配置
│   ├── model.py				// 模型
│   ├── train.py                // 模型训练
│   ├── predict.py         		// 模型预测
│   ├── Lenet.pth               // 权重文件
```

