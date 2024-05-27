# 수평선을 이용한 pitch 예측입니다만, 정확도가 너무 낮습니다.
# 따라서 최종적으로는 메인 코드에 사용하지 않았습니다.

import cv2
import numpy as np

def estimate_pitch(image):
    horizons = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Line Transform을 사용하여 선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 수평선 찾기
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y1 - y2) < 10:  # 수평선의 조건
                # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                horizon_y = (y1 + y2) / 2  # 수평선의 y 좌표
                horizons.append(horizon_y)

    # pitch 추정
    image_height = image.shape[0]
    pitch_angle = (np.average(horizons) - image_height / 2) / image_height * 90  # 간단한 비례 계산

    return pitch_angle, len(lines)