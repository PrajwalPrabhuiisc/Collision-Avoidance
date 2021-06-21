
import cv2

# Classifier used to detect the cars in video
cars_detect_cascade = cv2.CascadeClassifier('cars.xml')

# to draw the bounding boxes for cars
def detectingcars (videoframe):
    # To detect cars of varying magnitude
    cars = cars_detect_cascade.detectMultiScale(videoframe,1.2,5)
    # Draw boxes for all the cars detected
    for (x,y,w,h) in cars:
        cv2.rectangle(videoframe,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)
    return videoframe

def result():
    inp = cv2.VideoCapture('cars.mp4')
    while inp.isOpened():
        ret,frame = inp.read()
        controlkey = cv2.waitKey(1)
        if ret:
            cars_frame = detectingcars(frame)
            cv2.imshow("Result",cars_frame)
        else:
            break
        if controlkey == ord('q'):
            break
    inp.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    result()
