import cv2
from random import randrange

#Load some pretrained data on face frontals from opencv (haar cascade algorithm)
#cv2.CascadeClassifier class to detect object in a video stream / Classifier = Detector
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
#img = cv2.imread('faces.PNG')

#To campture video from webcam
webcam = cv2.VideoCapture(0)
#Listen for a key press for 1 millisecond, then move on
key = cv2.waitKey(1)

#Iterate forever over frames
while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    #detectMultiScale performs the detection
    face_coordinates = trained_face_data.detectMultiScale(frame)

    # Draw rectangles around the faces
    # for (x, y, w, h) in face_coordinates:
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # Show image
    cv2.imshow('Face Detection', frame)

    # Show image until the program is running
    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break
#Release the VideoCapture object
webcam.release()






#print(face_coordinates)


print("Code completed")



