import cv2
import argparse
from Retinaface.Retinaface import FaceDetector
from pathlib import Path

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--path', '-p', default='unknown', type=str, help='path of dir of person images')
args = parser.parse_args()
print('only a face in each image and all image from the same person')


dir_path = Path(args.path)
if not dir_path.is_dir():
    exit('dir does not exists !!')
save_path = Path(f'data/facebank/{dir_path.name}')
if not save_path.exists():
    save_path.mkdir()

# init detector
detector = FaceDetector(name='mobilenet', weight_path='Retinaface/weights/mobilenet.pth', device='cuda')

counter = 0
for img_path in dir_path.iterdir():
    img = cv2.imread(str(img_path))
    face = detector.detect_align(img)[0].cpu().numpy()
    if len(face.shape) > 1:
        save_name = f'{save_path}/{dir_path.name}_{counter}.jpg'
        cv2.imwrite(save_name, face[0])
        counter += 1
    else:
        print(img_path, 'in this image did not detect any face')