import tensorflow as tf, sys  
import matplotlib.image as img  
from picamera.array import PiRGBArray       
from picamera import PiCamera  
import RPi.GPIO as GPIO  
import time  
import cv2  
import numpy as np  
  
#hardware work  
GPIO.setmode(GPIO.BOARD)  
  
GPIO_TRIGGER1 = 29      #Left ultrasonic sensor  
GPIO_ECHO1 = 31  
  
GPIO_TRIGGER2 = 36      #Front ultrasonic sensor  
GPIO_ECHO2 = 37  
  
GPIO_TRIGGER3 = 33      #Right ultrasonic sensor  
GPIO_ECHO3 = 35  
  
MOTOR1B=22  #Left Motor  
MOTOR1E=18  
  
MOTOR2B=19  #Right Motor  
MOTOR2E=21  
  
LED_PIN=13    
  
# Set pins as output and input  
GPIO.setup(GPIO_TRIGGER1,GPIO.OUT)  # Trigger  
GPIO.setup(GPIO_ECHO1,GPIO.IN)      # Echo  
GPIO.setup(GPIO_TRIGGER2,GPIO.OUT)    
GPIO.setup(GPIO_ECHO2,GPIO.IN)  
GPIO.setup(GPIO_TRIGGER3,GPIO.OUT)    
GPIO.setup(GPIO_ECHO3,GPIO.IN)  
GPIO.setup(LED_PIN,GPIO.OUT)  
  
# Set trigger to False (Low)  
GPIO.output(GPIO_TRIGGER1, False)  
GPIO.output(GPIO_TRIGGER2, False)  
GPIO.output(GPIO_TRIGGER3, False)  
  
GPIO.setup(MOTOR1B, GPIO.OUT)  
GPIO.setup(MOTOR1E, GPIO.OUT)  
GPIO.setup(MOTOR2B, GPIO.OUT)  
GPIO.setup(MOTOR2E, GPIO.OUT)  
  
def sonar(GPIO_TRIGGER,GPIO_ECHO):  
      start=0  
      stop=0  
      # Set pins as output and input  
      GPIO.setup(GPIO_TRIGGER,GPIO.OUT)  # Trigger  
      GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo  
       
      # Set trigger to False (Low)  
      GPIO.output(GPIO_TRIGGER, False)  
       
      # Allow module to settle  
      time.sleep(0.01)  
             
      #while distance > 5:  
      #Send 10us pulse to trigger  
      GPIO.output(GPIO_TRIGGER, True)  
      time.sleep(0.00001)  
      GPIO.output(GPIO_TRIGGER, False)  
      begin = time.time()  
      while GPIO.input(GPIO_ECHO)==0 and time.time()<begin+0.05:  
            start = time.time()  
       
      while GPIO.input(GPIO_ECHO)==1 and time.time()<begin+0.1:  
            stop = time.time()  
       
      # Calculate pulse length  
      elapsed = stop-start  
      # Distance pulse travelled in that time is time  
      # multiplied by the speed of sound (cm/s)  
      distance = elapsed * 34000  
       
      # That was the distance there and back so halve the value  
      distance = distance / 2  
       
      print("Distance : %.1f" % distance)  
      # Reset GPIO settings  
      return distance  
  
#Function for alternate path  
def avoid():  
    stop()  
    leftturn()  
    time.sleep(0.8)  
    stop()  
    forward()  
    time.sleep(2)  
    stop()  
    rightturn()  
    time.sleep(0.7)  
    stop()  
    forward()  
    time.sleep(2)  
    stop()  
  
# Function to detect Obstacle      
def detect():  
    start = time.time()  
    time.sleep(0.5)  
    while(True):  
    distance = sonar(GPIO_TRIGGER2,GPIO_ECHO2)  
    if distance<15 and distance>8:  
        print('too close')  
        avoid()  
        return 0  
    now = time.time()  
    if (now - start) > 5:  
        print('time over')  
        stop()  
        return 0  
      
def forward():  
      GPIO.output(MOTOR1B, GPIO.HIGH)  
      GPIO.output(MOTOR1E, GPIO.LOW)  
      GPIO.output(MOTOR2B, GPIO.HIGH)  
      GPIO.output(MOTOR2E, GPIO.LOW)  
        
def rightturn():  
      GPIO.output(MOTOR1B,GPIO.HIGH)  
      GPIO.output(MOTOR1E,GPIO.LOW)  
      GPIO.output(MOTOR2B,GPIO.LOW)  
      GPIO.output(MOTOR2E,GPIO.HIGH)  
     
  
def reverse():  
      GPIO.output(MOTOR1B, GPIO.LOW)  
      GPIO.output(MOTOR1E, GPIO.HIGH)  
      GPIO.output(MOTOR2B, GPIO.LOW)  
      GPIO.output(MOTOR2E, GPIO.HIGH)  
   
        
def leftturn():  
      GPIO.output(MOTOR1B,GPIO.LOW)  
      GPIO.output(MOTOR1E,GPIO.HIGH)  
      GPIO.output(MOTOR2B,GPIO.HIGH)  
      GPIO.output(MOTOR2E,GPIO.LOW)  
        
def stop():  
      GPIO.output(MOTOR1E,GPIO.LOW)  
      GPIO.output(MOTOR1B,GPIO.LOW)  
      GPIO.output(MOTOR2E,GPIO.LOW)  
      GPIO.output(MOTOR2B,GPIO.LOW)  
  
  
dict={}  
#Function for Classification  
def classify(image_path):  
    # Read in the image_data  
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()  
    # Loads label file, strips off carriage return  
    label_lines = [line.rstrip() for line  
    in tf.gfile.GFile("retrained_labels.txt")]  
    # Unpersists graph from file  
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:  
    graph_def = tf.GraphDef()  
    graph_def.ParseFromString(f.read())  
    _ = tf.import_graph_def(graph_def, name='')  
    # Feed the image_data as input to the graph and get first prediction  
    score_max=0  
    ab=""  
    with tf.Session() as sess:  
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')  
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})  
            # Sort to show labels of first prediction in order of confidence  
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]  
            for node_id in top_k:  
                human_string = label_lines[node_id]  
                score = predictions[0][node_id]  
                human_string=str(human_string)  
                dict[human_string]=score  
                if score>score_max:  
                    ab=human_string  
                    score_max=score  
                #print('%s (score = %.5f)' % (human_string, score))  
  
    return (ab)  
      
  
#Getting a Camera Instance  
camera = PiCamera()  
camera.start_preview()  
time.sleep(5)  
camera.capture('/home/pi/btp4/image.jpg')  
camera.stop_preview()  
  
image_path = 'image.jpg'  
s=classify(image_path)  
print("cvlsidfvbslkdnfv"+s+"kanskuevfCSJECGVL")  
  
if s=="left" or s=="right":   
    image=img.imread(image_path)  
    im=np.rot90(image)  
    img_path=image_path+"2"  
    img.imsave(img_path,im)  
    abc=classify(img_path)  
    if abc=="up":  
    abcd="right"  
    accuracy=dict["up"]  
    else:  
    abcd="left"  
    accuracy=dict["down"]  
  
else:  
    abcd=s  
    accuracy=dict[s]  
  
print(str(abcd)+" with accuracy "+str(accuracy))  
s=abcd  
  
if s=="up":  
    reverse()  
    time.sleep(3)  
    stop()  
  
elif s=="down":  
    forward()  
    time.sleep(3)  
    stop()  
      
elif s=="left":  
    rightturn()  
    time.sleep(1.4)  
    stop()  
  
else:  
    leftturn()  
    time.sleep(1.4)  
    stop()  
      
GPIO.cleanup()  
