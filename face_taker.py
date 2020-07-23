import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Eenter user id end press <return> ==>  ')
face_name = input('\n Eenter user name end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w+50,y+h+50), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        gray = gray[y:y+h,x:x+w]
        
        cv2.imwrite("dataset/User." + str(face_name)+ '.' + str(face_id) + '.' + str(count) + ".jpg",gray )
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 30:
        break
    elif count >= 100: # Take 70 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


