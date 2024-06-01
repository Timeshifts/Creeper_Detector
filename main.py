import cv2, time, torch, datetime, pygame
from ultralytics import YOLO
from creeper import *
from option import *
from capture import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    if not use_example:
        capture_thread = threading.Thread(target=set_frame)
        capture_thread.start()
    record_thread = threading.Thread(target=record_frame)
    record_thread.start()

    global window_height, window_width

    pygame.init()
    pygame.mixer.init()

    # Pygame 창 설정
    pygame_screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Creeper Detector')

    icon = pygame.image.load('resource/icon.png')
    pygame.display.set_icon(icon)

    warn_sound = pygame.mixer.Sound('resource/warn.wav')
    shutter_sound = pygame.mixer.Sound('resource/shutter.wav')
    font = pygame.font.Font('resource/NanumBarunGothic.ttf', 48)
    video_file = 'resource/example_video.avi'

    model = YOLO('resource/creeper_model.pt')
    model.to(device)

    if use_example:
        video = cv2.VideoCapture(video_file)
        assert video.isOpened(), 'Cannot read the example video.'

    running = True
    show_help = False

    help_text = []
    help_string = ['Q: 도움말 호출', 'S: 화면 촬영', 'R: 크기 재조정', 'F9: 화면 녹화', 'ESC: 종료']

    for string in help_string:
        text = font.render(string, True, (0, 0, 0))
        help_text.append(text)

    while running and not stop_event.is_set():
        start_time = time.time()

        if use_example:
            vaild, frame = video.read()
            if not vaild:
                running = False
                break
        else:
            frame = get_frame(frame_queue)
        
        if frame is not None:
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

            if record_event.is_set():
                if not record_queue.full():
                    record_queue.put(frame)
                else:
                    # 큐가 꽉 찬 경우 가장 오래된 프레임을 버리고 새로운 프레임 추가
                    record_queue.get()
                    record_queue.put(frame)

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
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(f'screenshot/{timestamp}.png', frame)
                    print(f'screenshot/{timestamp}.png')
                elif event.key == pygame.K_r:
                    # 'r' 키로 화면 Resize
                    window_width, window_height = size
                    pygame_screen = pygame.display.set_mode(size)
                elif event.key == pygame.K_F9:
                    # F9 키로 화면 녹화
                    shutter_sound.play()
                    if record_event.is_set():
                        record_event.clear()
                    else:
                        record_event.set()
                elif event.key == pygame.K_q:
                    # q 키로 명령어 표시
                    show_help = not show_help

        if frame is not None:
            if record_event.is_set():
                pygame.draw.circle(pygame_screen, (255, 0, 0), (pygame_screen.get_width() - 50, 50), 30.)
                
        # 너무 빨리 처리된 경우 다음 캡처까지 대기
        # 실제 마인크래프트 화면을 사용할 경우 거의 일어나지 않습니다.
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)

        time.sleep(sleep_time)
        pygame.display.update()

    cv2.destroyAllWindows()
    pygame.quit()
    stop_event.set()
    capture_thread.join()
    record_thread.join()

if __name__ == '__main__':
    main()