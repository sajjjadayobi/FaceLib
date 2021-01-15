import os
import cv2
import torch
from facelib import FaceDetector
from pathlib import Path


def add_from_webcam(person_name='unknow', camera_index=0):
    print('loading ...')
    # create facebank folder if is not exists
    
    save_path = Path(os.path.dirname(os.path.realpath(__file__)), 'models/data/facebank')
    if not save_path.exists():
        save_path.mkdir()

    # create a new folder with (name) for new person
    save_path = save_path/person_name
    if not save_path.exists():
        save_path.mkdir()
    
    print('for exit: use q keyword')
    # init camera
    cap = cv2.VideoCapture(camera_index)
    cap.set(3, 1280)
    cap.set(4, 720)
    # init detector
    detector = FaceDetector()
    count = 4
    print('type q for exit')
    while cap.isOpened():
        ret , frame = cap.read()
        if ret == False:
            raise Exception('the camera not recognized: change camera_index param to ' + str(0 if camera_index == 1 else 1))
        
        frame = cv2.putText(
                frame, f'Press t to take {count} pictures', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,100,0), 3, cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            count -= 1
            faces = detector.detect_align(frame)[0].cpu().numpy()
            if len(faces.shape) > 1:
                cv2.imwrite(f'{save_path}/{person_name}_{count}.jpg', faces[0])
                if count <= 0:
                    break
            else:
                print('there is not face in this frame')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cv2.imshow("add new person from camera", frame)

    cap.release()
    cv2.destroyAllWindows()
    print('images saved in this path: ', save_path)



def add_from_folder(folder_path='./', person_name='unknow'):
    
    print('only a face in each image and all image from the same person')
    dir_path = Path(folder_path)
    if not dir_path.is_dir():
        exit('dir does not exists !!')

    # create facebank folder if is not exists
    save_path = Path(os.path.dirname(os.path.realpath(__file__)), 'models/data/facebank')
    if not save_path.exists():
        save_path.mkdir()


    save_path = Path(f'{save_path}/{person_name}')
    if not save_path.exists():
        save_path.mkdir()
    print('loading ...')
    # init detector
    detector = FaceDetector()

    counter = 0
    for img_path in dir_path.iterdir():
        img = cv2.imread(str(img_path))
        if img is None:
            raise Exception('this image has a problem: ', img_path)
        face = detector.detect_align(img)[0].cpu().numpy()
        if len(face.shape) > 1:
            save_name = f'{save_path}/{dir_path.name}_{counter}.jpg'
            cv2.imwrite(save_name, face[0])
            counter += 1
        else:
            print(img_path, 'in this image did not detect any face')
    print('images saved in this path: ', save_path)  
            
            
