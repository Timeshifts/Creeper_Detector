import cv2, time, torch, datetime, pygame, mss, mss.tools
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO
from creeper import *
from option import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sct = mss.mss()

frame_delay = 1 / 30

# Pygame 창 설정
pygame_screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Creeper Detector")

pygame.init()
pygame.mixer.init()

warn_sound = pygame.mixer.Sound('resource/warn.wav')
shutter_sound = pygame.mixer.Sound('resource/shutter.wav')
font = pygame.font.Font('resource/NanumBarunGothic.ttf', 48)
video_file = 'resource/example_video.avi'

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

    monitor = {"left": x, "top": y, "width": width, "height": height}
    
    # 화면 캡처
    frame = capture_screen(monitor)
    
    # OpenCV로 화면 크기 조정
    if resize:
        frame = cv2.resize(frame, (window_width, int(window_width * height / width)))

    return frame

def capture_screen(monitor=None):
    # 화면을 캡처하고 NumPy 배열로 변환
    screenshot = sct.grab(monitor)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
    return screenshot

def main():
    global pygame_screen, window_height, window_width

    model = YOLO('resource/creeper_model.pt')
    model.to(device)

    if use_example:
        video = cv2.VideoCapture(video_file)
        assert video.isOpened(), 'Cannot read the example video.'

    running = True
    recording = False
    target = None
    last_capture_time = time.time()
    show_help = False
    game_mode = False
    
    help_text = []
    help_string = ['Q: 도움말 호출', 'S: 화면 촬영', 'R: 크기 재조정', 'F9: 화면 녹화', 'ESC: 종료', 'G: 게임 모드']

    for string in help_string:
        text = font.render(string, True, (0, 0, 0))
        help_text.append(text)

    while running:
        start_time = time.time()

        if use_example:
            vaild, frame = video.read()
            if not vaild:
                running = False
                break
        else:
            frame = get_frame()
        warn_exists = find_creeper(model, frame)

        if warn_exists:
            if not pygame.mixer.get_busy(): warn_sound.play() 
        else:
            if pygame.mixer.get_busy(): warn_sound.stop()

        # OpenCV 프레임을 Pygame surface로 변환
        size = frame.shape[1::-1]
        pygame_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.image.frombuffer(pygame_frame.flatten(), size, 'RGB')
        pygame_frame = pygame.transform.scale(pygame_frame, (window_width, window_height))

        # Pygame 창에 OpenCV 프레임 표시
        pygame_screen.blit(pygame_frame, (0, 0))

        if show_help:
            y = 0
            for text in help_text: 
                pygame_screen.blit(text, (0, y))
                y += 64

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # 's' 키를 누르면 이미지 저장
                    shutter_sound.play()
                    # Generate a timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'screenshot/{timestamp}.png', frame)
                    print(f'screenshot/{timestamp}.png')
                elif event.key == pygame.K_r:
                    # 'r' 키로 화면 Resize
                    window_width, window_height = size
                    pygame_screen = pygame.display.set_mode(size)
                elif event.key == pygame.K_F9:
                    # F9 키로 화면 녹화
                    if recording: # REC -> no REC
                        target.release()
                    else: # no REC -> REC
                        target = cv2.VideoWriter()
                        # Open the target video file
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        target_file = f'screenshot/Record_{timestamp}.{record_format}'
                        is_color = (frame.ndim > 2) and (frame.shape[2] > 1)
                        last_capture_time = time.time()
                        target.open(target_file, cv2.VideoWriter_fourcc(*record_fourcc), 1 / frame_delay, size, is_color)
                    recording = not recording
                elif event.key == pygame.K_q:
                    # q 키로 명령어 표시
                    show_help = not show_help
                elif event.key == pygame.K_g:
                    # g 키로 게임 모드
                    game_mode = not game_mode

        # 화면 캡처 속도가 30fps에 못 미치므로,
        # 영상을 30프레임으로 보정
        elapsed_time_recording = time.time() - last_capture_time
        frames_to_add = int(elapsed_time_recording / frame_delay)

        if recording: 
            pygame.draw.circle(pygame_screen, (255, 0, 0), (pygame_screen.get_width() - 50, 50), 30.)
            print(frames_to_add)
            if frames_to_add > 0: 
                for _ in range(frames_to_add): target.write(frame)
            last_capture_time += frames_to_add * frame_delay
            
        # 너무 빨리 캡처된 경우 다음 캡처까지 대기
        # 실제 마인크래프트 화면을 사용할 경우 거의 일어나지 않습니다.
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)

        time.sleep(sleep_time)

        pygame.display.update()

    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()