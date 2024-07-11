from ultralytics import YOLO

# 载入pytorch模型
model = YOLO('m_best.pt')

# 导出模型
model.export(format='onnx')