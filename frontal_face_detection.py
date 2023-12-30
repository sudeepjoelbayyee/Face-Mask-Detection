import cv2

haar = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

def detect_face(img):
    coods = haar.detectMultiScale(img)
    return coods

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    coods = detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    for x,y,width,height in coods:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2 )

    cv2.imshow('window',frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()