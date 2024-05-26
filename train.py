from ultralytics import YOLO

def train():
    # 모델 로드
    model = YOLO('yolov8s.pt')

    results = model.train(data="creeper.yaml", epochs=3, imgsz=640, batch=8, auto_augment='randaugment')

    results = model.val()

    # Perform object detection on an image using the model
    # results = model("")

    # Export the model to ONNX format
    success = model.export(format="onnx")

if __name__ == '__main__':
    train()