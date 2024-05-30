from ultralytics import YOLO
import split_data as split_data

def train():
    # 모델 로드
    model = YOLO('yolov8s.pt')

    results = model.train(data="training/creeper.yaml", epochs=30, imgsz=640, batch=8, auto_augment='randaugment')

    results = model.val()

    # Export the model to ONNX format
    # success = model.export(format="onnx")

if __name__ == '__main__':
    split_data.split_data()
    train()