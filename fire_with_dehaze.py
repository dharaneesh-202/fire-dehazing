import cv2
import numpy as np
import smtplib
import threading

Alarm_Status = False
Email_Status = False
Fire_Reported = 0

def send_mail_function():
    recipientEmail = "bharath.2011010@srec.ac.in"
    recipientEmail = recipientEmail.lower()

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("bharathdws98424@gmail.com", '9842464576')
        server.sendmail('bharathdws98424@gmail.com', recipientEmail, "Warning: A Fire Accident has been reported ")
        print("sent to {}".format(recipientEmail))
        server.close()
    except Exception as e:
        print(e)

def dehaze(frame, t=0.1):
    I = frame / 255.0  # Normalize the frame
    dark_channel = np.min(I, axis=2)
    A = 1 - t * dark_channel

    A[A < 0.1] = 0.1  # Minimum value for A to avoid extreme dehazing
    
    J = (I - A[:, :, np.newaxis]) / np.maximum(A[:, :, np.newaxis], 0.1)
    J = (J * 255).astype(np.uint8)
    return J

def detect_fire(frame):
    frame = cv2.resize(frame, (960, 540))
    
    # Dehaze the frame
    dehazed_frame = dehaze(frame)

    blur = cv2.GaussianBlur(dehazed_frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    no_red = cv2.countNonZero(mask)

    if int(no_red) > 15000:
        # Find contours of the fire region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            for contour in contours:
                # Draw a bounding box around the fire region
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(dehazed_frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

                # Add text 'Fire'
                cv2.putText(dehazed_frame, 'Fire', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return True, dehazed_frame
    else:
        return False, dehazed_frame

video = cv2.VideoCapture("C:/Users/User/Downloads/fire/4.mp4")

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    fire_detected, frame = detect_fire(frame)

    if fire_detected:
        Fire_Reported += 1

    if Fire_Reported >= 1:
        if Alarm_Status == False:
            threading.Thread(target=send_mail_function).start()
            Alarm_Status = True

    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
