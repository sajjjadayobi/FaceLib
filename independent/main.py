# 基于facelib改写的独立使用的人脸识别
# 流程：判断需要检测对象的数据集，对其后存入facebank中

import os
import sys
from config import *
from Retinaface import *
from recognizer.FaceRecognizer import *
import cv2
from recognizer.model import *
from loguru import logger

# 将原始图像变为人脸图像 
# input_path保存需要识别对象的文件夹，结构：
# faceData
#   - nameA
#       - 1.jpg
#       - 2.jpg
#   - nameB
#       - 1.jpg
def data_align(detector,input_path = None,save_path=None):
    logger.info("data align start")
    if(input_path == None):
        input_path = os.path.dirname(os.path.realpath(sys.argv[0]))  + "\\faceData"
        logger.debug("input_path使用默认路径{}".format(input_path))
    if(save_path == None):
        save_path = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\data\\facebank"
        logger.debug("save_path使用默认路径{}".format(save_path))

    for root, dirs, files in os.walk(input_path, topdown=False):
        for name in dirs:
            dirs = os.listdir( os.path.join(root, name) )
            logger.info("start join people {}".format(name))
            for file in dirs:
                img_path = os.path.join(root, name, file)
                faceImg = cv2.imread(str(img_path))
                face = detector.detect_align(faceImg)[0].cpu().numpy()

                isExists = os.path.exists(save_path + "\\" + name)
                if not isExists:
                    os.makedirs(save_path + "\\" + name)
                    logger.debug("创建文件夹{}".format(save_path + "\\" + name))
                if len(face.shape) > 1:
                    savefile = save_path+"\\"+ name +"\\"+file
                    logger.info("{} img {} join success".format(name,file))
                    cv2.imwrite(savefile, face[0])
                else:
                    logger.info("{} img {} join fail, not find face".format(name,file))
                

# 更新比对数据库，产出facebank.pth和names.npy
def update_facebank(model,facebank_path=None,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    if(facebank_path == None):
        facebank_path = os.path.dirname(os.path.realpath(sys.argv[0]))  + "\\data\\facebank"
        logger.debug("识别库使用默认路径{}".format(facebank_path))

    if os.path.exists(facebank_path) == False:
        logger.error("not find facebank path")
        return None

    model.eval()
    faces_embs = torch.empty(0).to(device)
    names = np.array(['Unknown'])

    logger.info("start update facebank")
    for root, dirs, files in os.walk(facebank_path, topdown=False):
        faceList = []
        # 循环所有检测对象的文件夹
        for name in dirs:
            logger.info("start join people {}".format(name))
            dirs = os.listdir( os.path.join(root, name) )
            
            #循环文件夹中的所有图
            for file in dirs:
                img_path = os.path.join(root, name, file)
                faceImg = cv2.imread(str(img_path))
                if faceImg.shape[:2] != (112, 112):
                    logger.error("img {} not 112*112 , update is over".format(file))
                    return None
                else:
                    faceImg = torch.tensor(faceImg).unsqueeze(0)
                faceList.append(faceImg)

            faces = torch.cat(faceList)
            with torch.no_grad():
                faces = faces_preprocessing(faces, device=device)
                face_emb = model(faces)
                hflip_emb = model(faces.flip(-1))  # image horizontal flip
                face_embs = l2_norm(face_emb + hflip_emb)

            faces_embs = torch.cat((faces_embs, face_embs.mean(0, keepdim=True)))
            names = np.append(names, name)

    torch.save(faces_embs, facebank_path + '/facebank.pth')
    np.save(facebank_path +'/names', names)
    logger.info("update facebank success")
    return faces_embs, names

# 加载识别对象的库
def load_facebank(conf):
    if os.path.exists(conf.facebank_path) == False:
        logger.error("you don't have facebank yet: create with add_from_webcam or add_from_folder function")
        return None
    
    embs = torch.load(conf.facebank_path + '/facebank.pth')
    names = np.load(conf.facebank_path + '/names.npy')
    logger.info("load facebank success")
    return embs, names


import argparse

def main(args):

    conf = get_config()
    detector = FaceDetector(name= "resnet",device=conf.device,weight_path = conf.work_path)
    face_rec = FaceRecognizer(conf)# name="ir_se50"

    path = os.path.dirname(os.path.realpath(sys.argv[0])) 
    # 相关文件夹 
    if(os.path.exists(conf.work_path) == False):
        os.makedirs(conf.work_path)
    if(os.path.exists(conf.facebank_path) == False):
        os.makedirs(conf.facebank_path)
    if(os.path.exists(path + "/faceData") == False):
        os.makedirs(path + "/faceData")

    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m','--model', required=True ,choices=['find', 'update'])
    parser.add_argument('-i','--img', default='',help="input face image path")
    parser.add_argument('-o','--outImg', default='',help="output retinaface image path")
    args = parser.parse_args()

    if(args.model == "find"):
        targets, names = load_facebank(conf)
        if(targets == None):
            return
        imgPath = args.img
        outImgPath = args.outImg
        # 判断该图片是否存在
        if(os.path.exists(imgPath) == False):
            logger.warning("not find img {}".format(imgPath))
            return None
        image = cv2.imread(imgPath)
        faces, boxes, scores, landmarks = detector.detect_align(image)
        if(len(faces.shape) > 1):
            results, score = face_rec.infer(faces, targets,tta=True)  # return min_idx, minimum
            #print(results,score)
        else:
            logger.warning("not find face")
            sys.exit()
        for idx, bbox in enumerate(boxes):
            special_draw(image, bbox, landmarks[idx], names[results[idx]+1], score[idx])
            logger.info("name={},score={}".format(names[results[idx]+1],score[idx]))

            if(outImgPath == ""):
                cv2.imwrite("./out/{}.jpg".format(names[results[idx]+1]),image)
    elif(args.model == "update"):
        data_align(detector)
        update_facebank(face_rec.model)
    else:
        logger.warning("model error : {} ".format(args.model))
    return 


if __name__ == "__main__":
   main(sys.argv[1:])