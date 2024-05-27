import cv2, time, torch
import numpy as np
import pyautogui
import pygetwindow as gw
from ultralytics import YOLO
import datetime

def capture_screen(region=None):
    # 화면을 캡처하고 NumPy 배열로 변환
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

# OpenCV 창의 가로 크기
window_width = 1920
frame_delay = 1 / 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'

debug_mode = True

min_conf = 0.389
debug_conf = 0.1

def main():
    
    model = YOLO('creeper_model.pt')
    model.to(device)


    while True:
        windows = gw.getWindowsWithTitle('Minecraft')

        if len(windows) == 0:
            print('No Minecraft Window found!')
            cv2.destroyAllWindows()
            exit()

        # Find the Minecraft window
        minecraft_window = windows[0]

        # Get the coordinates and size of the Minecraft window
        x, y, width, height = minecraft_window.left, minecraft_window.top, minecraft_window.width, minecraft_window.height

        # 창 테두리 제거
        x += 10
        y += 60
        width -= 20
        height -= 70

        region = (x, y, width, height)

        start_time = time.time()

        # 화면 캡처
        frame = capture_screen(region)
        
        # OpenCV로 화면 크기 조정
        frame = cv2.resize(frame, (window_width, int(window_width * height / width)))

        # YOLOv5 모델로 객체 탐지
        results = model(frame, verbose=False)
        
        # 결과 처리
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                if conf > min_conf:  # 신뢰도가 일정 값 이상일 때만 표시
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.circle(frame, (int((x1 + x2) / 2), int(y2)), 15, (0, 0, 0), -1)
                if debug_mode:
                    if conf > min_conf:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    elif conf > debug_conf:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        

        cv2.imshow('Creeper Detector', frame)

        key = cv2.waitKey(1)

        # 'ESC' 키를 누르면 루프 종료
        if key & 0xFF == 27:
            break
        # 's' 키를 누르면 이미지 저장
        elif key & 0xFF == ord('s'):
            # Generate a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'screenshot/{timestamp}.png', frame)
            print(f'screenshot/{timestamp}.png')

        # 다음 캡처까지 대기
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()