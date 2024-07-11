from ultralytics import YOLO
import torch

import cv2
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

######################################################
# 载入模型
model = YOLO('H:/Cache/PyThon的储存/机器学习/apple4/Train_apple/s_pretrain/weights/best.pt')
# model = YOLO('H:/Cache/PyThon的储存/机器学习/apple4/Train_apple_3/m_pretrain/weights/m_best.pt')
######################################################

# 切换计算设备
model.to(device)
# model.cpu()  # CPU
# model.cuda() # GPU

print("模型自带信息")
print(model.device)



#预测
#传入图像、视频、摄像头ID（对应命令行的 source 参数）
########################################################
img_path = 'H:/Cache/PyThon的储存/机器学习/apple4/test_/apple3.jpg'
########################################################
results = model(img_path)

#解析预测结果
len(results)
results[0]

print("解析目标检测预测结果:")
print("1.预测框的所有类别")
print(results[0].names)
print("2.预测类别 ID")
print(results[0].boxes.cls)
num_bbox = len(results[0].boxes.cls)
print('3.预测出 {} 个框'.format(num_bbox))
print("4.每个框的置信度")
print(results[0].boxes.conf)
print("5.每个框的：左上角XY坐标、右下角XY坐标")
print(results[0].boxes.xyxy)
print("6.转成整数的 numpy array")
bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
print(bboxes_xyxy)

# OpenCV可视化关键点
img_bgr = cv2.imread(img_path)
# #原始图像
# plt.imshow(img_bgr[:,:,::-1])
# plt.show()

# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽



for i in range(num_bbox):
    # 获取当前框的坐标
    xyxy = bboxes_xyxy[i]
    # 绘制框
    cv2.rectangle(img_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), bbox_color, bbox_thickness)

    # 获取类别名称和置信度
    cls_id = int(results[0].boxes.cls[i])
    cls_name = results[0].names[cls_id]
    conf = results[0].boxes.conf[i]

    # 设置标签文本
    cls_text = '{}: {:.2f}%'.format(cls_name, conf * 100)

    # 获取标签的尺寸
    text_size = cv2.getTextSize(cls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]

    # 计算标签的位置
    text_x = xyxy[0]
    text_y = xyxy[1] - 5 if xyxy[1] - 5 > 10 else xyxy[1] + 5

    # 绘制标签背景
    cv2.rectangle(img_bgr, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0] + 2, text_y + 2), bbox_color, -1)

    # 绘制文本
    cv2.putText(img_bgr, cls_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

plt.imshow(img_bgr[:,:,::-1])
plt.show()
# # 保存带有检测框的图像
# cv2.imwrite('C1_output_with_boxes.jpg', img_bgr)