import cv2, time, torch, pyautogui
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO
import datetime, pygame, math

# from pitch_estimation import estimate_pitch
from pose_estimation import draw_ar

# OpenCV 창의 가로 크기
window_width = 1920
frame_delay = 1 / 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_options_from_file(file_path, encoding='UTF-8'):
    options = {}
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=')
                options[key.strip()] = value.strip()
    return options

options = read_options_from_file('options.txt')

debug_mode = options.get('debug_mode', 'false').lower() == 'true'
pose_estimation = options.get('pose_estimation', 'false').lower() == 'true'

min_conf = float(options.get('min_conf', 0.389))
debug_conf = float(options.get('debug_conf', 0.1))

warning_distance = float(options.get('warning_distance', 7))

pygame.init()
pygame.mixer.init()
warn_sound = pygame.mixer.Sound('resource/warn.wav')

def get_frame(resize=True):
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
    y += 50
    width -= 20
    height -= 60

    region = (x, y, width, height)
    
    # 화면 캡처
    frame = capture_screen(region)
    
    # OpenCV로 화면 크기 조정
    if resize:
        frame = cv2.resize(frame, (window_width, int(window_width * height / width)))

    return frame

def capture_screen(region=None):
    # 화면을 캡처하고 NumPy 배열로 변환
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

def draw_explosion(frame, box, pitch=10):
    height, width = frame.shape[:2]  # 이미지의 높이와 너비 추출
    x1, y1, x2, y2 = box
    
    # 플레이어의 각도에 따라 값이 크게 변하지 않는 너비를 기준으로 사용
    creeper_real = (0.6, 1.7)  # 실제 크리퍼의 너비, 높이 (블록 단위)
    creeper_width_pixels = x2 - x1  # 감지된 크리퍼의 너비 (픽셀 단위)
    creeper_height_pixels = creeper_width_pixels / creeper_real[0] * creeper_real[1]
    
    # pitch_estimation.py 참고
    # estimated_pitch, count = estimate_pitch(frame)
    # pitch = np.clip(estimated_pitch, 10.0, 40.0)

    # 카메라에서 크리퍼까지의 거리 계산
    # 마인크래프트의 fov는 수직 각도만이 설정된 값으로 고정되므로, 크리퍼의 높이 사용
    fov = 70
    distance = (creeper_real[1] * height) / (2 * creeper_height_pixels * np.tan(np.deg2rad(fov / 2)))
    distance = distance ** 1.125
    if debug_mode: print(f'{distance} {(y2 - y1) / (x2 - x1)}')

    explosion_radius_blocks = 3  # 크리퍼 폭발 반경 (블록 단위)
    explosion_radius_pixels_x = int((explosion_radius_blocks / creeper_real[0]) * creeper_width_pixels)
    explosion_radius_pixels_y = int((height - y2) / distance * explosion_radius_blocks * 5 / 8)
    warn_radius_blocks = 7  # 크리퍼 폭발 취소 반경 (블록 단위)
    warn_radius_pixels_x = int((warn_radius_blocks / creeper_real[0]) * creeper_width_pixels)
    warn_radius_pixels_y = int((height - y2) / distance * warn_radius_blocks * 12 / 21)

    # 크리퍼 발 좌표 계산
    explosion_center = (int((x1 + x2) / 2), int(y2 + explosion_radius_pixels_y * 3 / 8))
    warn_center = (int((x1 + x2) / 2), int(y2 + warn_radius_pixels_y * 11 / 21))

    alpha = 0.15  # 투명도 값 (0: 완전 투명, 1: 완전 불투명)
    
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)  # 빈 이미지 생성
    # 폭발 반경 표시
    cv2.ellipse(blank_image, explosion_center, 
                (explosion_radius_pixels_x, explosion_radius_pixels_y),
                0, 0, 360, (0, 0, 255), -1)
    # 이미지 합성 (반투명 효과)
    cv2.addWeighted(blank_image, alpha, frame, 1, 0, frame)
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)  # 빈 이미지 생성
    cv2.ellipse(blank_image, warn_center, 
                (warn_radius_pixels_x, warn_radius_pixels_y),
                0, 0, 360, (0, 255, 255), -1)
    cv2.addWeighted(blank_image, alpha, frame, 1, 0, frame)

    return distance

def main():
    
    model = YOLO('resource/creeper_model.pt')
    model.to(device)

    while True:
        start_time = time.time()

        frame = get_frame()
        frame_2 = frame.copy()
        # YOLOv8 모델로 객체 탐지
        results = model(frame, verbose=False)
        
        # 결과 처리
        warn_exists = False
        
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                if conf > min_conf:  # 신뢰도가 일정 값 이상일 때만 표시
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    distance = draw_explosion(frame, box)
                    if distance < warning_distance:
                        warn_exists = True
                    # pose_estimation.py 참고
                    if pose_estimation:
                        creeper_image = frame_2[int(y1):int(y1+(x2-x1)*1.2), int(x1):int(x2)]
                        creeper_image = draw_ar(creeper_image)
                        if creeper_image is not None:
                            cv2.imshow('Another Detector', creeper_image)
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                    if conf > min_conf:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f'{distance:.2f} m', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if debug_mode:
                        cv2.putText(frame, label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        if conf < min_conf and conf > debug_conf:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)      

        if warn_exists:
            if not pygame.mixer.get_busy(): warn_sound.play() 
        else:
            if pygame.mixer.get_busy(): warn_sound.stop()

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