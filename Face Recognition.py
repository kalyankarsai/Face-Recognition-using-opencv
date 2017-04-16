import os
import cv2
import sqlite3
import numpy as np
from PIL import Image
from flask import Flask



app = Flask(__name__)


def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT* FROM People WHERE ID="+str(Id)
    isRecordExists=0
    cursor=conn.execute(cmd)
    for row in cursor:
        isRecordExists=1
    if(isRecordExists==1):
        cmd="UPDATE People SET Name="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

@app.route('/dataset')
def dataset():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)
    id=raw_input('enter user id')
    name=raw_input('enter your name')
    insertOrUpdate(id,name)
    p=0
    while(True):
        ret,img=cam.read()
        grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(grey,1.3,5)
        for(x,y,w,h) in faces:
            p=p+1
            cv2.imwrite("dataSet/User."+str(id)+"."+str(p)+".jpg",grey[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("Face",img)
        cv2.waitKey(1)
        if(p>20):
            break
    cam.release()
    cv2.destroyAllWindows()
    return 'Done adding data !!'


def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


@app.route('/detector')
def detect():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    cam=cv2.VideoCapture(0);
    rec=cv2.createLBPHFaceRecognizer();
    rec.load("recognizer/trainningData.yml")
    id=0
    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
    #time.sleep(1)
    ret, img = cam.read()
    #Checking if cam is opened
    cap=cam.isOpened()
    #print cap,ret
    if(not ret):
        return 'camera not initialised'
    while(True):
        ret,img=cam.read();
        grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(grey,1.3,5);
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            id,conf=rec.predict(grey[y:y+h,x:x+w])
            profile=getProfile(id)
            if(profile!=None):
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]),(x,y+h),font,255)
        cv2.imshow("Face",img)
        if(cv2.waitKey(1)==ord('q')):
            break;
    cam.release()
    cv2.destroyAllWindows()
    return 'Done Detecting !!'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("trainning",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces


@app.route('/trainer')
def train():
    recognizer=cv2.createLBPHFaceRecognizer();
    path='dataSet'
    Ids,faces=getImagesWithID(path)
    recognizer.train(faces,Ids)
    recognizer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()
    return 'Done Training !!'

        
if __name__ == '__main__':
   app.run(host='0.0.0.0',port='5005')
