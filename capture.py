import datetime, cv2, mss, mss.tools, queue, threading, time
import numpy as np
import pygetwindow as gw
from option import *


frame_delay = 1 / 30
frame_queue = queue.Queue(maxsize=10)
record_queue = queue.Queue(maxsize=10)

stop_event = threading.Event()
record_event = threading.Event()

def capture_screen(sct):
    
    windows = gw.getWindowsWithTitle('Minecraft')

    if len(windows) == 0:
        print('No Minecraft Window found!')
        stop_event.set()
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

    monitor = {"left": x, "top": y, "width": width, "height": height}
    
    # 화면을 캡처하고 NumPy 배열로 변환
    frame = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGRA2BGR)

    # OpenCV로 화면 크기 조정
    frame = cv2.resize(frame, (window_width, int(window_width * height / width)))

    return frame
    
def set_frame():
    sct = mss.mss()

    while not stop_event.is_set():
        start_time = time.time()
        # 화면 캡처
        frame = capture_screen(sct)
        if not frame_queue.full():
                frame_queue.put(frame)
        else:
            # 큐가 꽉 찬 경우 가장 오래된 프레임을 버리고 새로운 프레임 추가
            frame_queue.get()
            frame_queue.put(frame)
            
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)

        time.sleep(sleep_time)

def record_frame():
    Recording = False
    frame = None
    while not stop_event.is_set():
        if record_event.is_set():
            frame = get_frame(record_queue, False)

            if not Recording: # no REC -> REC
                Recording = True
                target = cv2.VideoWriter()
                # Open the target video file
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                target_file = f'screenshot/Record_{timestamp}.{record_format}'
                is_color = (frame.ndim > 2) and (frame.shape[2] > 1)
                last_record_time = time.time()
                size = frame.shape[1::-1]
                target.open(target_file, cv2.VideoWriter_fourcc(*record_fourcc), 1 / frame_delay, size, is_color)

            # 화면 캡처 속도가 30fps에 못 미치므로,
            # 영상을 30프레임으로 보정
            elapsed_time_recording = time.time() - last_record_time
            frames_to_add = int(elapsed_time_recording / frame_delay)

            print(frames_to_add)
            if frames_to_add > 0: 
                for _ in range(frames_to_add): target.write(frame)
            last_record_time += frames_to_add * frame_delay
        else:
            if Recording: # REC -> no REC
                print('release')
                target.release()
                Recording = False
            time.sleep(0.01)

def get_frame(target_queue: queue.Queue, return_none=True):
    while True:
        if not target_queue.empty():
            # 큐에서 가장 최신 프레임만 가져오기
            while not target_queue.empty():
                frame = target_queue.get()
            return frame
        else:
            # 큐가 비어있으면 빈 반환
            if return_none:
                return None
            else:
                time.sleep(0.01)