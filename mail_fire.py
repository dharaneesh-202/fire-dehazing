import cv2
import numpy as np
import smtplib
import threading
import geocoder
from datetime import datetime

# Global variables
Alarm_Status = False
Email_Status = False
Fire_Reported = 0

# Function to send an email with incident location and screenshot
def send_mail_function(lat, lon, screenshot_path):
    recipientEmail = "madhan.2105054@srec.ac.in"
    recipientEmail = recipientEmail.lower()

    # Get the device's current location
    g = geocoder.ip('me')
    location = g.latlng  # Get latitude and longitude

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("madhan.2105054@srec.ac.in", 'madhan2004')

        # Construct the email message with location information
        message = f"Warning: A Fire Accident has been reported at the following location:\nLatitude: {location[0]}\nLongitude: {location[1]}"

        # Attach the screenshot to the email
        with open(screenshot_path, 'rb') as screenshot_file:
            screenshot_data = screenshot_file.read()

        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg.attach(MIMEText(message, 'plain'))
        
        image = MIMEImage(screenshot_data, name="screenshot.jpg")
        msg.attach(image)

        # Set email subject and recipients
        msg['Subject'] = "Fire Report"
        msg['From'] = "madhan.2105054@srec.ac.in"
        msg['To'] = recipientEmail

        # Send the email
        server.sendmail('madhan.2105054@srec.ac.in', recipientEmail, msg.as_string())
        print("Sent to {}".format(recipientEmail))
        server.close()
    except Exception as e:
        print(e)

# Function to dehaze the frame
def dehaze(frame, t=0.1):
    I = frame / 255.0  # Normalize the frame
    dark_channel = np.min(I, axis=2)
    A = 1 - t * dark_channel

    A[A < 0.1] = 0.1  # Minimum value for A to avoid extreme dehazing
    
    J = (I - A[:, :, np.newaxis]) / np.maximum(A[:, :, np.newaxis], 0.1)
    J = (J * 255).astype(np.uint8)
    return J

# Function to detect fire and draw bounding boxes
def detect_fire(frame):
    frame = cv2.resize(frame, (960, 540))
    
    # Dehaze the frame (you can use the dehaze function from a previous response)
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

# Capture video from a file or camera (change the source as needed)
video = cv2.VideoCapture("C:/Users/User/Downloads/fire/4.mp4")  # Replace with your video source

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    fire_detected, frame = detect_fire(frame)

    if fire_detected:
        Fire_Reported += 1

    if Fire_Reported >= 1:
        if Alarm_Status == False:
            # Get and pass the current location to the send_mail_function
            g = geocoder.ip('me')
            location = g.latlng

            # Capture a screenshot of the current frame
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            screenshot_path = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)

            # Send the email with the screenshot
            threading.Thread(target=send_mail_function, args=(location[0], location[1], screenshot_path)).start()
            Alarm_Status = True

    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
