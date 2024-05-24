import cv2, time
import numpy as np
import pyautogui
import pygetwindow as gw

def capture_screen(region=None):
    # 화면을 캡처하고 NumPy 배열로 변환
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

# OpenCV 창의 가로 크기
window_width = 1920

def main():
    print(cv2.__version__)
    print(np.__version__)
    print(pyautogui.__version__)
    print(gw.__version__)
    frame_delay = 1 / 30

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
        region = (x, y, width, height)

        start_time = time.time()

        # 화면 캡처
        frame = capture_screen(region)
        
        # OpenCV로 화면 처리
        frame_resized = cv2.resize(frame, (window_width, int(window_width * height / width)))
        cv2.imshow('MC Thing', frame_resized)
        
        # 'ESC' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # 다음 캡처까지 대기
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()