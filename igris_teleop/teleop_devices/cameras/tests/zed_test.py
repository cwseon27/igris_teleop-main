import cv2

dev = "/dev/video0"  # 실제 생성된 번호로 바꾸세요 (/dev/video2 등)
cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {dev}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("read failed")
        break

    # SBS(좌우 한 프레임)일 가능성 있으면 아래처럼 분리해서 확인
    h, w = frame.shape[:2]
    if w % 2 == 0:
        left = frame[:, :w//2]
        right = frame[:, w//2:]
        cv2.imshow("left", left)
        cv2.imshow("right", right)
    else:
        cv2.imshow("frame", frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
