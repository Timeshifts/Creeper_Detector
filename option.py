def read_options_from_file(file_path, encoding='UTF-8'):
    options = {}
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=')
                options[key.strip()] = value.strip()
    return options

options = read_options_from_file('options.txt')

use_example = options.get('use_example', 'false').lower() == 'true'

debug_mode = options.get('debug_mode', 'false').lower() == 'true'
pose_estimation = options.get('pose_estimation', 'false').lower() == 'true'

min_conf = float(options.get('min_conf', 0.389))
debug_conf = float(options.get('debug_conf', 0.1))

warning_distance = float(options.get('warning_distance', 7))
assert warning_distance > 0, 'Warning distance must be positive.'

# 창 관련 정보
window_width = int(options.get('window_width', 1920))
window_height = int(options.get('window_height', 1080))
assert window_width > 640, 'Too small window width!'
assert window_height > 480, 'Too small window height!'

record_format = options.get('record_format', 'avi')
record_fourcc = options.get('record_format', 'XVID')