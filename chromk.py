import cv2
import numpy as np


start = True
pressed = False
start1 = 0
start2 = 1
start3 = 1
maxVal1 = 30
maxVal2 = 10
maxVal3 = 8
softVal = 1
hue = 10
kernelSize = 1
back = 0




def mouse(action,x,y,flags,userdata):
    global pressed, back, lower_color, upper_color
    if action == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        back = hsvFrame[(y,x)][0]
        lower_color = np.array([back-hue, 50, 50], dtype = "uint8")
        upper_color = np.array([back+hue, 255, 255], dtype = "uint8")

def tolerance(*args):
    global hue, lower_color,upper_color
    hue = args[0]
    lower_color = np.array([back-hue, 50, 30], dtype = "uint8")
    upper_color = np.array([back+hue, 255, 255], dtype = "uint8")

def softness(*args):
    global softVal
    softVal = args[0]
    if softVal%2 == 0:
        softVal = softVal + 1

def colorCast(*args):
    global kernelSize, element
    kernelSize = args[0]
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernelSize+1,
    2*kernelSize+1),(kernelSize, kernelSize))



cv2.namedWindow('Video',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Tolerance', 'Video', start1, maxVal1, tolerance)
cv2.createTrackbar('Softness', 'Video', start2, maxVal2, softness)
cv2.createTrackbar('Color cast', 'Video', start3, maxVal3, colorCast)
cv2.setMouseCallback('Video', mouse)


cap = cv2.VideoCapture('greenscreen-asteroid.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
height = int(cap.get (cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get (cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.avi',
cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))

img = cv2.imread('background.jpg')
img = cv2.resize(img, (width,height), interpolation= cv2.INTER_LINEAR)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if start == True:
            cv2.putText(frame, 'Click on the background', (20, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1.5,(250, 10, 10), 2, cv2.LINE_AA);
            cv2.imshow('Video',frame)
            colorCast(1)
            start = False

            while pressed != True:
                cv2.waitKey(25)

        thresh = 255 - cv2.inRange(hsvFrame, lower_color, upper_color)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,element,iterations=1)
        alpha = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2RGB)
        alpha = alpha.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha,(softVal,softVal),0)
        newFrame = frame * alpha + img * (1-alpha)
        newFrame = newFrame.astype(np.uint8)
        cv2.imshow('Video',newFrame)
        out.write(newFrame)
        if cv2.waitKey(25) & 0xFF == 27:
            break

    else:
        cap.release()
        break

f = cv2.waitKey()
if f == 27:
    cv2.destroyAllWindows()
