import argparse
import cv2
from openvino.inference_engine import IECore, IENetwork
import platform
import math

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
    tl_desc = "A threshold above which the bounding box will be yello, default 0.85"
    #color of the bounding boxes, for the lower accuracy
    c_desc = "Define the colour of the bounding boxes. Choose from 'RED', 'GREEN', BLUE', default 'RED' "
    

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default=0.5)
    optional.add_argument("-tl", help=tl_desc, default=0.85)
    optional.add_argument("-c", help=c_desc, default="RED")
    args = parser.parse_args()

    return args

def preprocessing(frame, width, height):
    '''
    Preprocess the image to fit the model.
    '''
    frame = cv2.resize(frame, (width, height))
    frame = frame.transpose((2,0,1))
    return frame.reshape(1, 3, width, height)

def draw_tracked_vehicles(frame, output, threshold, threshold_limit, width, height, box_color, tracked_vehicles):
    '''
    Draw a bounding box in the tracked objects
    '''

    for tracked in output[0][0]:
        #only if the accuracy is higher than the threshold
        if tracked[2]>threshold:
            #get the centroid of each tracked vehicle
            cent = ((((tracked[3] + tracked[5])*width)/2).round(5), ((tracked[4] + tracked[6])*height/2).round(5))
            #print(cent)
            
            #append only the non zero coordinates
            if cent != (0.0, 0.0) :
                #if it is the 1st entry in the tracked_vehicles, just append
                if len(tracked_vehicles)==0:
                    tracked_vehicles.append(cent)
                else:
                    for coord in tracked_vehicles:
                        #calculate how close this prediction is along the oces that are already calculated
                        #print(len(tracked_vehicles))
                        x_dist = cent[0]-coord[0]
                        y_dist = cent[1]-coord[1]
                        dist = math.sqrt(x_dist**2 + y_dist**2)
                        #print(dist)
                        
                    #append if distance of centroid is smaller that 0.5 
                    if dist>35:
                        tracked_vehicles.append(cent)
                    else:
                        print("found stopped vehicle")
                        #we remove previous entry and only keep the last one
                        idx = tracked_vehicles.index((coord[0], coord[1]))
                        del tracked_vehicles[idx]
                        tracked_vehicles.append(cent)
                        #get the conner coordinates of each tracked vehicle
                        x1 = int(tracked[3] * width)
                        y1 = int(tracked[4] * height)
                        x2 = int(tracked[5] * width)
                        y2 = int(tracked[6] * height)
                        #draw a bounding box based on them 
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)
                        
    return frame
        

def draw_boxes(frame, output, threshold, threshold_limit, width, height, box_color):
    '''
    Draws the colored bounding boxes in the detected objects.
    '''
    if box_color.lower() == "blue":
        color = (255,0,0)
    elif box_color.lower() == "green":
        color = (0,255,0)
    else:
        color = (0,0,255)

    #print(output.shape) #during debug to get the shape of the frame
    for fr in output[0][0]:
        if fr[2]>threshold:
            
            x1 = int(fr[3] * width)
            y1 = int(fr[4] * height)
            x2 = int(fr[5] * width)
            y2 = int(fr[6] * height)
            #if the accuracy is above the limit specified, then the rectangle will be yellow
            if fr[2]>=threshold_limit:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
    return frame

def perform_inference(network, exec_network, args, request_id):
    #for the given network, calculate
    input_blob = get_input_blob(network=network)
    output_blob = get_output_blob(network=network)
    input_shape = get_input_shape(network=network, input_blob=input_blob)
    if DEBUG:
        print(input_blob)
        print(output_blob)
        print(input_shape)
#    ### TODO: Initialize the Inference Engine
#    ie = Network()
#    ### TODO: Load the network model into the IE
#    ie.load_model(model=args.m, device=args.d, cpu_extension=CPU_EXTENSION)
#    net_input_shape = ie.get_input_shape()
#    #print(net_input_shape)
    #Load the modle 
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    if DEBUG:
        print(height, width)
    
    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'MPG4')
    out = cv2.VideoWriter('out.mp4', fourcc, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    # placeholder for the coordinates of the centroid of each object tracked by the algorithm
    centr_coords = []
    # in here we are gonna track the coordinates of the stopped vehicles
    tracked_vehicles = []

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
            #frame = draw_boxes(frame=frame, output=out_frame, threshold=float(args.t), threshold_limit=float(args.tl), width=width, height=height, box_color=args.c)
            frame = draw_tracked_vehicles(frame=frame, output=out_frame, threshold=float(args.t), threshold_limit=float(args.tl), width=width, height=height, box_color=args.c, tracked_vehicles=tracked_vehicles)

            #print("len of tracked vehicles", len(tracked_vehicles))
            ##every 15 frames, check if there are stopped vehicles tracked
            #while len(tracked_vehicles) >= 15:
            #    del tracked_vehicles[0]

            #print("tracked vehicles:", tracked_vehicles)



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
    if len(unsup_layers) == 1:
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
    if not missing_layers:
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
 

    #infer_on_video(args)


if __name__ == "__main__":
    main()
