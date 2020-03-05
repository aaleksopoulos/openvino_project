import argparse
import cv2
from openvino.inference_engine import IECore, IENetwork
import platform
import math
from tracked_cars import Tracked_Cars

DEBUG = False #dummy variable

if (platform.system() == 'Windows'):
    CPU_EXTENSION = "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\\bin\intel64\Release\cpu_extension_avx2.dll"
else:
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"

    #confidence thresholds used to draw bounding boxes
    t_desc = "The threshold for model accuracy, default 0.5"
    #color of the bounding boxes, for the lower accuracy
    c_desc = "Define the colour of the bounding boxes. Choose from 'YELLOW', 'GREEN', BLUE', default 'YELLOW' "
    

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default=0.5)
    optional.add_argument("-c", help=c_desc, default="YELLOW")
    args = parser.parse_args()

    return args

def preprocessing(frame, width, height):
    '''
    Preprocess the image to fit the model.
    '''
    frame = cv2.resize(frame, (width, height))
    frame = frame.transpose((2,0,1))
    return frame.reshape(1, 3, width, height)

def track_objects_iou(frame, tracked_vehicles, current_tracked_centroids, current_tracked_box_coord, box_color, carId, checkStopped=False):
    '''
    Tracks the car objects in the frame, returns the last carId found
    If checkStopped is False, it will try to track objects, else it will try to track stopped objects
    '''
    #placeholder for all vehicles tracked in current frame, plus the ones of the previous. Will replace tracked_vehicle
    car_list = [] 
    if DEBUG:
        print("len of tracked centroids: ", len(current_tracked_centroids))
        print("len of tracked vehicles: ", len(tracked_vehicles))
        print("carId: ", carId)
    #if it is the 1st frame calculated, just append it to the list
    if len(tracked_vehicles) ==0:
   
        for i in range(len(current_tracked_box_coord)):
            centroid = current_tracked_centroids[i]
            box = current_tracked_box_coord[i]
            #register a new car
            car = Tracked_Cars(carId=carId, centroid=centroid, x1=box[0], x2=box[1], y1=box[2], y2=box[3])
            #append the car to the car list and increase the index
            car_list.append(car)
            carId += 1
            
            #print it to the frame
            if not checkStopped:
                cv2.putText(frame, car.toString(), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
                cv2.rectangle(frame, (car.getX1(),car.getY1() ), (car.getX2(),car.getY2() ), box_color, 1)

    else:
        #check the cars that were tracked in the previous frame
        #for tracked in tracked_vehicles[-1]:
        for tracked in tracked_vehicles:   
            #placeholder to track the iou
            ious = []

            #get the coordinates and the area of each tracked object
            trackedX1 = tracked.getX1()
            trackedX2 = tracked.getX2()
            trackedY1 = tracked.getY1()
            trackedY2 = tracked.getY2()
            trackedArea = tracked.getArea()
            
            for i in range(len(current_tracked_box_coord)):
                #get the coordinates of each tracked car in current frame
                curX1 = current_tracked_box_coord[i][0]
                curY1 = current_tracked_box_coord[i][1]
                curX2 = current_tracked_box_coord[i][2]
                curY2 = current_tracked_box_coord[i][3]
                cur_area = abs(curX1 - curX2) * abs(curY1 - curY2)
                #calculate the iou for each, if there is an overlap
                if (((curX1>trackedX1 and curX1<trackedX2) or (curX2>trackedX1 and curX2<trackedX2)) and ((curY1>trackedY1 and curY1<trackedY2) or (curY2>trackedY1 and curY2<trackedY2))):
                    
                    #iou = areaOfOverlap/areaOfUnion
                    #get the coordinates of the intesection square
                    #a list to hold the x and y-coordinates
                    x = [trackedX1, trackedX2, curX1, curX2]
                    y = [trackedY1, trackedY2, curY1, curY2]
                    #the intersection area will be from the inbetween coordinates
                    x.sort()
                    y.sort()
                    interArea = (x[2]-x[1]) * (y[2]-y[1])

                    iou = (interArea) / (cur_area + trackedArea - interArea)
                    if DEBUG:
                        print("----------------------------------------------------------------------------------------------------")
                        print("interArea: ", interArea)
                        print("current area: ", cur_area)
                        print("tracked area: ", trackedArea)
                        print("box of current area: x1: ", curX1 , ' x2: ', curX2, " y1: ", curY1, " y2: ", curY2)
                        print("box of tracked area: x1: ", trackedX1 , ' x2: ', trackedX2, " y1: ", trackedY1, " y2: ", trackedY2)
                        print('x coords: ', x)
                        print('y coords: ', y)
                        print('iou: ', iou)
                        print("----------------------------------------------------------------------------------------------------")
                    ious.append(iou)
                
            #if any iou was calculated
            if(len(ious)!=0):
                #get the max iou
                max_iou = max(ious)
                if DEBUG:
                    print(ious)
                    print(max_iou)
                max_idx = ious.index(max_iou)
            
                #get the coordinates of the box  
                x1 = current_tracked_box_coord[max_idx][0]
                y1 = current_tracked_box_coord[max_idx][1]
                x2 = current_tracked_box_coord[max_idx][2]
                y2 = current_tracked_box_coord[max_idx][3]
                centroid = current_tracked_centroids[max_idx]
                if (max_iou)>=0.30 and not checkStopped:
                    #update the coordinates fo the box
                    tracked.setX1(x1)
                    tracked.setX2(x2)
                    tracked.setY1(y1)
                    tracked.setY2(y2)
                    tracked.setCentroid(centroid)
                    car_list.append(tracked)
                    #remove the car from the current lists
                    current_tracked_centroids.remove(centroid)
                    current_tracked_box_coord.remove(current_tracked_box_coord[max_idx])
                    #tracked_vehicles[-1].remove(tracked)
                    tracked_vehicles.remove(tracked)
                    #put the box on the frame and the car id
                    cv2.rectangle(frame, (curX1,curY1), (curX2,curY2), box_color, 1)
                    cv2.putText(frame, tracked.toString(), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
                    
                elif (max_iou)>=0.92 and checkStopped:
                    #update the coordinates fo the box
                    tracked.setX1(x1)
                    tracked.setX2(x2)
                    tracked.setY1(y1)
                    tracked.setY2(y2)
                    tracked.setCentroid(centroid)
                    car_list.append(tracked)
                    #remove the car from the current lists
                    current_tracked_centroids.remove(centroid)
                    current_tracked_box_coord.remove(current_tracked_box_coord[max_idx])
                    #tracked_vehicles[-1].remove(tracked)
                    tracked_vehicles.remove(tracked)
                    #put the box on the frame
                    cv2.rectangle(frame, (curX1,curY1), (curX2,curY2), box_color, 1)

        #add everything left as a new object
        for i in range(len(current_tracked_box_coord)):
            #get the box coordinates
            centroid = current_tracked_centroids[i]
            x1 = current_tracked_box_coord[i][0]
            y1 = current_tracked_box_coord[i][1]
            x2 = current_tracked_box_coord[i][2]
            y2 = current_tracked_box_coord[i][3] 
            car = Tracked_Cars(carId=carId, centroid=centroid, x1=x1, x2=x2, y1=y1, y2=y2)
            car_list.append(car)
            carId += 1
            #print(car.toString())
            if not checkStopped:
                cv2.rectangle(frame, (car.getX1(),car.getY1() ), (car.getX2(),car.getY2() ), box_color, 1)
                cv2.putText(frame, car.toString(), car.getCentroid(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
        if DEBUG:
            print("len of car_list before: ", len(car_list))
            print("len of tracked: ", len(tracked_vehicles))
        
        #add all the remaining tracked vehicles to the car_list, updating their attributes
        #for tracked in tracked_vehicles[-1]:
        for tracked in tracked_vehicles:
            #track how many times it was disappeared
            if not tracked.getTracked():
                tracked.setDisappearedFames(tracked.getDisappearedFrames() + 1)
            else:
                tracked.setDisappearedFames(0)
            #set tracked status to false
            tracked.setTracked(False)
            #add only if it has not been disappeared for more than maxDisappearedFrames
            if tracked.getDisappearedFrames() <= tracked.maxDisappearedFrames:
                car_list.append(tracked)
        if DEBUG:
            print("len of car_list after: ", len(car_list))   
    #return the last carId calcualted, and the car_list to be used for the next frame
    return carId, car_list

def track_objects(frame, tracked_vehicles, current_tracked_centroids, current_tracked_box_coord, box_color, carId, minDist=0, checkStopped=False):
    '''
    Tracks the car objects in the frame, returns the last carId found
    If checkStopped is False, it will try to track objects, else it will try to track stopped objects
    In that case a minDist not equal to zero should be specified
    '''
    car_list = [] 
    #print("len of tracked centroids: ", len(current_tracked_centroids))
    #print("len of tracked vehicles: ", len(tracked_vehicles))
    #print("carId: ", carId)
    #if it is the 1st frame calculated, just append it to the list
    if len(tracked_vehicles) ==0:
   
        for i in range(len(current_tracked_centroids)):
            centroid = current_tracked_centroids[i]
            x1 = current_tracked_box_coord[i][0]
            y1 = current_tracked_box_coord[i][1]
            x2 = current_tracked_box_coord[i][2]
            y2 = current_tracked_box_coord[i][3]
            car = Tracked_Cars(carId=carId, centroid=centroid, x1=x1, x2=x2, y1=y1, y2=y2)
            #append the car to the car list and icrease the index
            car_list.append(car)
            carId += 1
            #print it to the frame
            if not checkStopped:
                cv2.putText(frame, car.toString(), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)

    else:

        #check for the cars that were tracked in the last frame
        #for tracked in tracked_vehicles[-1]:
        for tracked in tracked_vehicles:
            #placeholder to track the distances
            cent_dist = []
            tracked_centroid = tracked.getCentroid()
            
            for i in range(len(current_tracked_centroids)):
                centroid = current_tracked_centroids[i]
                #calculate the dist from the current tracked cars
                dist = math.sqrt(math.pow((centroid[0]-tracked_centroid[0]),2) + math.pow((centroid[1]-tracked_centroid[1]),2))
                #print(dist)
                cent_dist.append(dist)
                
                
            #if any distance was calculated
            if(len(cent_dist)!=0):
                #get the min distance and its index
                min_dist = min(cent_dist)
                
                if (min_dist<=minDist) and not checkStopped:
                    min_idx = cent_dist.index(min_dist)
                    #print("centroid distances: ", cent_dist)
                    #print('min idx:', min_idx)

                    #set the new cetroid and add this one to the new car list
                    tracked.setCentroid(current_tracked_centroids[min_idx])
                    car_list.append(tracked)
                    #remove the car from the current list
                    current_tracked_centroids.remove(centroid)
                    #tracked_list.remove(tracked)
                    #tracked_vehicles[-1].remove(tracked)
                    tracked_vehicles.remove(tracked)
                    x1 = current_tracked_box_coord[i][0]
                    y1 = current_tracked_box_coord[i][1]
                    x2 = current_tracked_box_coord[i][2]
                    y2 = current_tracked_box_coord[i][3]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 1)
                    cv2.putText(frame, tracked.toString(), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
                elif (min_dist<=2) and checkStopped:

                    min_idx = cent_dist.index(min_dist)
                    #print("centroid distances: ", cent_dist)
                    #print('min idx:', min_idx)

                    #set the new cetroid and add this one to the new car list
                    tracked.setCentroid(current_tracked_centroids[min_idx])
                    car_list.append(tracked)
                    #remove the car from the current list
                    current_tracked_centroids.remove(centroid)
                    #tracked_list.remove(tracked)
                    #tracked_vehicles[-1].remove(tracked)
                    tracked_vehicles.remove(tracked)
                    #cv2.putText(frame, tracked.toString(), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
                    x1 = current_tracked_box_coord[i][0]
                    y1 = current_tracked_box_coord[i][1]
                    x2 = current_tracked_box_coord[i][2]
                    y2 = current_tracked_box_coord[i][3] 
                    cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 1)


    #add everything left as a new object
    for i in range(len(current_tracked_centroids)):
        leftovers = current_tracked_centroids[i]
        #print("leftovers: ", leftovers)
        x1 = current_tracked_box_coord[i][0]
        y1 = current_tracked_box_coord[i][1]
        x2 = current_tracked_box_coord[i][2]
        y2 = current_tracked_box_coord[i][3]
        car = Tracked_Cars(carId=carId, centroid=leftovers, x1=x1, x2=x2, y1=y1, y2=y2)
        car_list.append(car)
        carId += 1
        if not checkStopped:
 
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 1)
            cv2.putText(frame, car.toString(), car.getCentroid(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)

    return carId, car_list

def draw_boxes(frame, output, threshold, width, height, box_color, carId, tracked_vehicles):
    '''
    Draws the colored bounding boxes in the detected objects.
    '''
    if box_color.lower() == "blue":
        color = (255,0,0)
    elif box_color.lower() == "green":
        color = (0,255,0)
    else:
        color = (0,255,255)

    #placeholder for the tracked centroids and boxes
    current_tracked_centroids = []
    current_tracked_box_coord = []
    #print(output.shape) #during debug to get the shape of the frame
    for fr in output[0][0]:
        if fr[2]>threshold:
            #calculate the coordinates of the bounding box of the tracked car
            x1 = int(fr[3] * width)
            y1 = int(fr[4] * height)
            x2 = int(fr[5] * width)
            y2 = int(fr[6] * height)
            #calculate the centroid of the tracked car
            centroid = ((x1+x2)//2, (y1+y2)//2)
            box_coord = (x1, y1, x2 ,y2)
            #append it to the lists
            current_tracked_centroids.append(centroid)
            current_tracked_box_coord.append(box_coord)

    #track the objects found in the new frame, based on the previous
    #carId, tracked_vehicles = track_objects(frame=frame, tracked_vehicles=tracked_vehicles, current_tracked_centroids=current_tracked_centroids, current_tracked_box_coord=current_tracked_box_coord, box_color=color, carId=carId, minDist=12, checkStopped=True)
    carId, tracked_vehicles = track_objects_iou(frame=frame, tracked_vehicles=tracked_vehicles, current_tracked_centroids=current_tracked_centroids, current_tracked_box_coord=current_tracked_box_coord, box_color=color, carId=carId, checkStopped=True)    
    return carId, frame, tracked_vehicles

def perform_inference(network, exec_network, args, request_id):
    #for the given network, calculate
    input_blob = get_input_blob(network=network)
    output_blob = get_output_blob(network=network)
    input_shape = get_input_shape(network=network, input_blob=input_blob)
    if DEBUG:
        print(input_blob)
        print(output_blob)
        print(input_shape)
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    if DEBUG:
        print(height, width)
    
    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('out.mp4', fourcc, 30, (width,height))
    
    #placeholder for the different cars we have found
    tracked_vehicles = []
    carId = 0 #placeholder for the carIds to be tracked
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        prep_frame = preprocessing(frame, input_shape[2], input_shape[3])
        
        # Perform inference on the frame
        exec_network.start_async(request_id=request_id, inputs={input_blob: prep_frame})
        
        # Get the output of inference
        if exec_network.requests[request_id].wait(-1)==0:
            out_frame = exec_network.requests[request_id].outputs[output_blob]
            if DEBUG:
                print(out_frame)
            
            # Update the frame to include detected bounding boxes
            carId, frame, tracked_vehicles = draw_boxes(frame=frame, output=out_frame, threshold=float(args.t), width=width, height=height, box_color=args.c, carId=carId, tracked_vehicles=tracked_vehicles)

        # Write out the frame
        out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def get_input_blob(network):
    return next(iter(network.inputs))

def get_output_blob(network):
    return next(iter(network.outputs))

def get_input_shape(network, input_blob):
    return network.inputs[input_blob].shape

def check_unsupported_layers(ie, network_model, device_name):
    '''
    Given an Inference engine, network model and device name it will
    return True if there are unsupported layers, and False if all
    layers are supported
    '''
    layers = ie.query_network(network=network_model, device_name=device_name)
    if DEBUG:
        print("printing supported layers")
        print(layers)
        print("===========================================")

    #get a list of the required layers
    req_layers = list(network_model.layers.keys())
    #get a list of the supported layers
    sup_layers = list(layers.keys())
    #initiaze an empty list to hold the unsuporrted layers
    unsup_layers = []
    #check if we are missing any layer and add it to the list
    for layer in req_layers:
        if layer not in sup_layers:
            unsup_layers.append(layer)
    if DEBUG:
        print("printing unsupported layers")
        print(unsup_layers)
        print("===========================================")
    #return False if all layers are supported, True otherwise
    if len(unsup_layers) == 0:
        return False
    else:
        return True

def load_model_to_IE(args):
    #get the location of model xml and bin files
    model_xml = args.m
    model_bin = model_xml.replace(".xml", ".bin") 
    #Load the Inference Engine API
    iec = IECore()
    #Load IR files into their related class
    if DEBUG:
        print(model_xml)
        print(model_bin)
    #create the network
    ien = IENetwork(model=model_xml, weights=model_bin)
        
    #check if there are layers unsupported
    missing_layers =  check_unsupported_layers(ie=iec, network_model=ien, device_name=args.d)
    
    if missing_layers and args.d=="CPU":
        try:
            iec.add_extension(extension_path=CPU_EXTENSION, device_name="CPU")
        except:
            #in openvino 2020 there are no CPU extensions
            print("something went wrong reading the cpu extension, exiting") 
            exit(1)
    #now we are gonna recheck if there are missing layers, and exit if there are 
    missing_layers =  check_unsupported_layers(ie=iec, network_model=ien, device_name=args.d)
    if missing_layers:
        print("after adding CPU extension there are still unsupported layers, exiting")
        exit(1)
   
    #Load the network into the Inference Engine
    exec_network = iec.load_network(network=ien, device_name=args.d)
    
    return ien, exec_network

def main():
    args = get_args()
    request_id = 0
  
    network, exec_network = load_model_to_IE(args=args)
    perform_inference(network= network, exec_network=exec_network, args=args, request_id=request_id)

if __name__ == "__main__":
    main()
