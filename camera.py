import cv2

# Replace these with your actual values
username = "admin"
password = "admin"
ip = "192.168.1.64"  # Camera's IP on your Wi-Fi network
channel = 1           # 1 = main stream, 2 = sub stream
stream = 1            # 1 = primary, 2 = secondary

rtsp_url = f"rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel}0{stream}"

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read from camera.")
        break
    cv2.imshow("Hikvision Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
