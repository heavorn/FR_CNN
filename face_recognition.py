import numpy as np
import cv2
import os
import h5py
import dlib
# from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Dense, Activation, Flatten
from PIL import Image
from Model import model

# def getImagesAndLabels():
    
#     path = 'dataset2'
#     imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
#     faceSamples=[]
#     ids = []

#     for imagePath in imagePaths:

#         #if there is an error saving any jpegs
#         try:
#             PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
#         except:
#             continue  
#         img_numpy = np.array(PIL_img,'uint8')
#         id = int(os.path.split(imagePath)[-1].split(".")[1])
#         faceSamples.append(img_numpy)
#         ids.append(id)
#     return faceSamples,ids
    

# _,ids = getImagesAndLabels()


def getImagesAndLabels():
    
    path = 'dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    labels = {}

    for imagePath in imagePaths:

        #if there is an error saving any jpegs
        try:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        except:
            continue  
        img_numpy = np.array(PIL_img,'uint8')

        label = str(os.path.split(imagePath)[-1].split(".")[1])
        id = int(os.path.split(imagePath)[-1].split(".")[2])
        labels[label] = id
        faceSamples.append(img_numpy)
        ids.append(id)
    return faceSamples,ids,labels
    

_,ids,labels_dic = getImagesAndLabels()
sorted_label = sorted(labels_dic.items())
labels = []
for i in range(len(sorted_label)):
    labels.append(sorted_label[i][0])

model = model((32,32,1),len(set(ids)))
model.load_weights  ('trained_model_testing.h5')
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX
def start():
    cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)
    # Min Height and Width for the  window size to be recognized as a face
    # minW = 0.1 * cap.get(3)
    # minH = 0.1 * cap.get(4)
    ret = True

    # clip = []
    while ret:
        #read frame by frame
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        # nframe = frame
        faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30))

        # try:
        #     (x,y,w,h) = faces[0]
        # except:
        #     continue
        # frame = frame[y:y+h,x:x+w]
        # frame = cv2.resize(frame, (32,32))
        
        # c= cv2.waitKey(1)
        # if c & 0xFF == ord('q'):
        #     break

        # labels = ['Ell', 'Sovorn', 'Sambo']

        #gray = gray[np.newaxis,:,:,np.newaxis]
        
        
        for (x, y, w, h) in faces:
            gray = cv2.cvtColor(cv2.resize(frame, (32,32)), cv2.COLOR_BGR2GRAY)
            gray = gray.reshape(-1, 32, 32, 1).astype('float32') / 255.
            prediction = model.predict(gray)
            prediction = prediction.tolist()
            listv = prediction[0]
            n = listv.index(max(listv))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(labels[n]), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(frame, str(round(max(listv), 3)), (x+5,y+h-5), font, 1, (255,255,0), 1)
        # for (x, y, w, h) in faces:
        #     try:
        #         gray = gray.reshape(-1, 32, 32, 1).astype('float32') / 255.
        #         prediction = model.predict(gray)
        #         prediction = prediction.tolist()
        #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #         cv2.putText(frame, str(labels[n]), (x+5,y-5), font, 1, (255,255,255), 2)
        #         cv2.putText(frame, str(round(max(listv), 3)), (x+5,y+h-5), font, 1, (255,255,0), 1)
        #     except:
        #         la = 2 
        # prediction = np.argmax(model.predict(gray), 1)
        # # print(prediction)
        # cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('result', frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
start()
