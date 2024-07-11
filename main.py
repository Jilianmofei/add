import multiprocessing
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 训练模型
if __name__ == '__main__':
    multiprocessing.freeze_support()  # 只在 Windows 上需要
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolov8m.yaml")  # build a new model from scratch
    # Use the model
    model.train(data="apple.yaml", epochs=100, pretrained=True, project="Train_apple_m", name="m_pretrain", batch=16, device=0)  # train the model
