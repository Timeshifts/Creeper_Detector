import cv2
import numpy as np
from option import *
from pose_estimation import draw_ar

def draw_explosion(frame, box):
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

def find_creeper(model, frame):

    if pose_estimation: frame_2 = frame.copy()

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
    
    return warn_exists