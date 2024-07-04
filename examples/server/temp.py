import cv2
from ultralytics import YOLOv10
cap = cv2.VideoCapture(0)


model = YOLOv10.from_pretrained(f'jameslahm/yolov10n')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape)
    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    results = model.predict(source=frame, imgsz=640, conf=0.25)
    annotated_frame = results[0].plot()
    # Display the resulting frame
    cv2.imshow('Video Capture', annotated_frame)

        # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()