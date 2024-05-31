import cv2, time, torch, pyautogui, datetime, pygame
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO
from creeper import *
from option import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

frame_delay = 1 / 30

# Pygame 창 설정
pygame_screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Creeper Detector")

pygame.init()
pygame.mixer.init()

warn_sound = pygame.mixer.Sound('resource/warn.wav')
shutter_sound = pygame.mixer.Sound('resource/shutter.wav')
font = pygame.font.Font('resource/NanumBarunGothic.ttf', 24)

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

def main():
    
    model = YOLO('resource/creeper_model.pt')
    model.to(device)

    running = True
    while running:
        start_time = time.time()

        frame = get_frame()
              
        warn_exists = find_creeper(model, frame)

        if warn_exists:
            if not pygame.mixer.get_busy(): warn_sound.play() 
        else:
            if pygame.mixer.get_busy(): warn_sound.stop()

        # OpenCV 프레임을 Pygame surface로 변환
        size = frame.shape[1::-1]
        pygame_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pygame_frame = np.rot90(pygame_frame)
        pygame_frame = pygame.image.frombuffer(pygame_frame.flatten(), size, 'RGB')
        pygame_frame = pygame.transform.scale(pygame_frame, (window_width, window_height))

        # Pygame 창에 OpenCV 프레임 표시
        pygame_screen.blit(pygame_frame, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # 's' 키를 누르면 이미지 저장
                    # Generate a timestamp
                    shutter_sound.play()
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'screenshot/{timestamp}.png', frame)
                    print(f'screenshot/{timestamp}.png')

        # 다음 캡처까지 대기
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)

        pygame.display.update()

    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()