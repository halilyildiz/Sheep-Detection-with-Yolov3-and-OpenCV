# Import libraries
import cv2
import numpy as np
import time

# Import trained model
net = cv2.dnn.readNet('yolov3//yolov3.weights','yolov3//yolov3.cfg')

# Capture video
cap = cv2.VideoCapture('input-video.mp4')

# Video size
width = int(cap.get(3))
height = int(cap.get(4))

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output-video1.mp4',fourcc, 10, (width,height))

# Detect Box Style
font = cv2.FONT_HERSHEY_SIMPLEX
color = (60,60,60)
label = "Sheep"

# Counter
count = 0

# Start Time
start_time = time.time()

# Sheep detection process
while True:
    # Take a frame from the video
    ret, frame = cap.read()
    
    # Is there frame?
    if ret == False:
        print("End")
        break

    # Convert blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255,(608,608),(0,0,0),swapRB=True, crop = False)

    # Model input and forward
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    # Decompose outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Confidence Threshold and Filter Sheep
            if confidence > 0.5:
                if class_id == 18:            
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))

    # Show the detected sheep in the box
    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    
    # Draw boxes
    if type(indices) == type(np.array([])):
        for i in indices.flatten():
            x,y,w,h = boxes[i]

            cv2.rectangle(frame,(x,y),(x+w, y+h),color,2)
            cv2.rectangle(frame,(x,y-45),(x+110, y),color,thickness=-1)
            cv2.putText(frame,label, (x+5,y-15), font, 1, (255,255,255),2)
    
    # Frame written
    out.write(frame)
    count += 1
    processed_second = round(count/30,1)
    elapsed_time = round((time.time() - start_time)/60,1)

    print("Processed Frame: {} || Processed Second: {} sec. || Elapsed Time: {} min.".format(count,processed_second, elapsed_time))
    cv2.imshow("Video",frame)
    
    #key = cv2.waitKey(0)
    #if key == 27:
        #break
cap.release()
cv2.destroyAllWindows() 

