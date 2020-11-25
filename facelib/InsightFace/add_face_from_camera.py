import cv2
import argparse
from Retinaface.Retinaface import FaceDetector
from pathlib import Path

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='unknown', type=str, help='input the name of the recording person')
args = parser.parse_args()

save_path = Path('data/facebank')/args.name
if not save_path.exists():
    save_path.mkdir()

# init camera
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
# init detector
detector = FaceDetector(name='resnet', weight_path='Retinaface/weights/resnet50.pth', device='cuda')
count = 4
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.putText(
                frame, f'Press t to take {count} pictures, then finish...', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,100,0), 3, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        count -= 1
        faces = detector.detect_align(frame)[0].cpu().numpy()
        if len(faces.shape) > 1:
            cv2.imwrite(f'{save_path}/{args.name}_{count}.jpg', faces[0])
            if count <= 0:
                break
        else:
            print('there is not face in this frame')

    cv2.imshow("My Capture", frame)

cap.release()
cv2.destroyAllWindows()
