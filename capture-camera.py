import cv2
import os

dataset_path = "dataset"
class_name = "class1"

def get_camera() -> cv2.VideoCapture:
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap

if __name__ == '__main__':
    directory = dataset_path
    count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]) + 1
    cap = get_camera()
    print(f"Starting capture image from camera, starting from {count}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Collecting Data", frame)

        # กด 's' เพื่อบันทึกภาพ
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"dataset/{class_name}_{count}.jpg", frame)
            count += 1
            print(f"Saved image {count}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()