import cv2
import random
import cvzone


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_eye_data = cv2.CascadeClassifier('haarcascade_eye.xml')

beard_filter = cv2.imread('beard.png', cv2.IMREAD_UNCHANGED)

#lets capture the video from the local camera
vid = cv2.VideoCapture(0)

while True:
    #read the frame
    ret, frame = vid.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray_frame)
    eyes_coordinates = trained_eye_data.detectMultiScale(gray_frame)

 
    
    for (x, y, w, h) in face_coordinates:
        print(x,y,w,h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (random.randrange(256), random.randrange(256), random.randrange(256)), 4)
        
        #resizing the filter to the size of the current face detected
        beard_filter = cv2.resize(beard_filter, (int(w*1.5), int(h*1.5)))

     
        frame = cvzone.overlayPNG(frame, beard_filter, [x-45, y-70])



    cv2.imshow('Snap filter Clone', frame)
    if cv2.waitKey(1) & 0xFF  == ord('x'):
        break


vid.release()
cv2.destroyAllWindows()