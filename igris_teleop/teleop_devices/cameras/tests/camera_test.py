import pyrealsense2 as rs

# 1) 연결된 디바이스 정보 출력
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print(">>> ERROR: 현재 연결된 RealSense 디바이스가 없습니다.")
    print(">>> 카메라를 연결했는지, USB 케이블 상태와 리얼센스 드라이버 설치 여부를 확인하세요.")
    exit(1)
else:
    print(">>> 연결된 RealSense 디바이스 개수:", len(devices))
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  디바이스 #{i + 1}")
        print(f"    - 이름(Name)       : {name}")
        print(f"    - 시리얼 넘버(SN)  : {serial}")
        print(f"    - 펌웨어 버전(FW) : {fw}")

# 2) Pipeline과 Config 생성
pipeline = rs.pipeline()
config = rs.config()

# 3) 활성화할 스트림 정보 기록 & 로그 출력
stream_type = rs.stream.color
width, height, fmt, fps = 1280, 720, rs.format.bgr8, 30
config.enable_stream(stream_type, width, height, fmt, fps)

print("\n>>> Config 객체에 아래 스트림을 설정했습니다:")
print(f"    스트림 종류: {stream_type} (rs.stream.color)")
print(f"    해상도    : {width}×{height}")
print(f"    포맷      : {fmt} (rs.format.bgr8)")
print(f"    프레임레이트: {fps}fps\n")

# 4) 실제로 파이프라인 시작
try:
    profile = pipeline.start(config)
    print(">>> 카메라 연결 성공! 파이프라인이 시작되었습니다.")
    # (필요하다면 profile.get_stream() 등을 통해 세부 스트림 정보를 꺼낼 수 있음)
except RuntimeError as e:
    print(">>> RuntimeError 발생:", str(e))
    print(">>> 카메라 연결에 실패했습니다. 원인 파악을 위해 아래를 확인하세요:")
    print("   - 실제로 RealSense 카메라가 USB에 꽂혀 있는지?")
    print("   - Linux(Ubuntu)라면 udev 규칙, 권한 설정이 제대로 되어 있는지?")
    print("   - Librealsense2가 제대로 설치되어 있는지 (예: apt-get 상태 등)?")
    print("   - 다른 프로세스(예: rs-enumerate-devices 등)에서 이미 카메라를 점유하고 있지 않은지?")
    exit(1)

# (추가 동작: 예를 들어 프레임 받아서 화면에 출력 등)

