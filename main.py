import cv2
import numpy as np
import face_recognition

imgRajini=face_recognition.load_image_file('imagebasics/old rajini.jpg')
imgRajini=cv2.cvtColor(imgRajini,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('imagebasics/rajinikanth-7593.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgRajini)[0]
encodeRajini=face_recognition.face_encodings(imgRajini)[0]
cv2.rectangle(imgRajini,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeRajini],encodeTest)
facedis=face_recognition.face_distance([encodeRajini],encodeTest)
print(results,facedis)
cv2.putText(imgTest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)

cv2.imshow('Rajini Kanth',imgRajini)
cv2.imshow('Rajini Kanth new',imgTest)
cv2.waitKey(0)
