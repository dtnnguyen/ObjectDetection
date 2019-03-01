#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:26:32 2018

@author: Trang N.
"""

#Object Detection Using Class method

#torch: library for computer vision that contains dynamich graphs
#to compute very efficiently gradient discent and backward propagation.
#Dynamic graphs: allows very fast and efficient computation of the gradients

import torch

# Variable: contains both the tensor and gradient will be one element of the graph.
from torch.autograd import Variable

# To drawing rectangles
import cv2

#Class BaseTransform will perform the required transformations so that 
#the input image will be compatible with the neural network.
#VOC_CLASSES: will do the incoding of the classifications. 
# ie: Dog = 1, planes = 2 ==> its a dictionary of classification.
from data import BaseTransform, VOC_CLASSES as labelmap

#build_ssd: is the constructor of ssd (single shot detection)
from ssd import build_ssd

#imageio: the library will process images of video
import imageio

class CObjectDetection(object):
    ###### setup()
    def setup(self, modelName):
        #Create the SSD neural network and name it test to test the model
        self.m_net = build_ssd(modelName)

        #torch.load open the tensor that will contain the object model weights
        #load_state_dictattribute(): atrribute the weights to the neural network.
        self.m_net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc:storage))

        #Creating the transformation
        #net.size = is the target size of the image to feed to the neural network
        #2nd parameter: tuple of colour scale that that the neural network was trained
        self.m_transform = BaseTransform(self.m_net.size, (104/256, 117/256, 123/256))
    #end setup()

    #initialize function ~ constructor
    def __init__(self, modelName):
        #Create the SSD neural network and name it test to test the model
        self.setup(modelName)
    #end of constructor

    # Name: detect()
    # Description: FRAME by FRAME detection
    #       SSD will be performed per image that imageIO extracted from all frames of video.
    #       Then resample the frames to display the video with the rectangles.
    # Paramter:
    #       frame: input image from video, doesn't need to be gray image.
    #       net: SSD neural network
    #       transform: transformation applied to the image =>
    #                       convert image into the neural network format.
    # Return:
    #       return image with rectangles and label
    def detect(self, frame, net, transform):
        #width and height of the image
        height, width = frame.shape[:2] #range 2 for width and height
        
        #FIRST TRANSFORMATION
        #   frame_t = transformed frame
        #   transform function return 2 elements, but we only want the first element
        #   [0] = first return element = transformed frame with the right format for NN
        #return numpy array
        frame_t = transform(frame)[0]
        
        #SECOND TRANSFORMATION = torch tensor 
        #   transform the frame into torch tensor, which is a much more advanced matrix 
        #   of a single type than an array.
        #   torch variables: dynamic graph
        #   x will be converted from a numpy array into a torch tensor.
        #   then convert rbg (0,1,2) into grb(2,0,1) ==> .permute(2,0,1)
        x = torch.from_numpy(frame_t).permute(2,0,1)
        
        #THIRD TRANSFORMATION (unsqueeze())
        #   Neural network can only accept batches of inputs;
        #   Thus, need to convert data into neural network format.
        #   Always starts witht the first index at 0th = first dimension
        
        #FOURTH TRANSFORMATION:(Variable()) ==> torch variable contains gradient and tensor.
        #   Torch variable will become an elelment of the dynamic graph
        #   which will compute very efficiently the gradients of any composition functions
        #   during backward propagation.
        #   unsqueeze(0) parameter is the index of the dimension of the batch 
        x = Variable(x.unsqueeze(0))
        
        #FIFTH: Send Input into neural net.
        y = net(x)
        
        #SIXTH: retreive tensor
        #   A tensor of 4-dimension: width, height, width, height
        #   Extract the neural network output:
        detections = y.data 
        
        #The first width, height ~ scale value of the top left corner.
        #The second width, height ~ scale value of the bottom right corner.
        #Position of an object inside an image needs to be normalized between 0 and 1.
        scale = torch.Tensor([width, height, width, height])
        
        # parameters of torch tensor: 
        # detections = [batch created from unsqueeze(), number of classes = objects, 
        #               number of occurence for the task(ie: dog), 
        #               tuple(score, x0, y0, x1, y1) = coordinates of the rectangles]
        for i in range(detections.size(1)):
            #  j:the occurence of the class
            j = 0
            #[the batch index 0, occurence j of the class i, score index 0
            while detections[0,i,j,0] > 0.6:
                #keep the occurence if the result is > 0.6
                # pt = extract the coordinate where the occurence of the class is found.
                #1: ==> index 1 of the detection tuple to the end
                #use scale tensor to apply normalization which will give the coordinates of 
                #these points at the scale of the image.
                #(detections[0, i, j, 1:]*scale): is a tensor
                #numpy(): to convert the rectangle coordinates into numpy array.
                pt = (detections[0, i, j, 1:]*scale).numpy()
                                
                #draw rectangles with the 4 coordintes from pt
                #2nd last argument: color of the rectangle
                #last argument: thichness of the rectangle
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)
                
                #print label labelmap[i-1] = name of the class
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), font, 2, (255,255,255), 2, cv2.LINE_AA)             
                j += 1       
        return frame
    
    #Doing some object detection on a video
    def start(self,inputVideoName, outputVideoName):
        #open video 
        reader = imageio.get_reader(inputVideoName)
        
        #get frequent of the video
        fps = reader.get_meta_data()['fps']
        
        #create writer to create new video with labels 
        writer = imageio.get_writer(outputVideoName, fps = fps)
        
        #step thru images of the video
        for i, frame in enumerate(reader):
            #net.eval() get detections from each frame
            frame = self.detect(frame, self.m_net.eval(), self.m_transform)
            #append the frame to the output.mp4
            writer.append_data(frame)
            print(i)
            
        #complete writer
        writer.close()
    #end of start()

#Proceed
#Enter input and output video names here
inputVideoName = 'IMG_WhiteEar.mp4'
outputVideoName = 'OutputIMG_WhiteEar.mp4'

# Create an instance with test mode, then call detection method
obj = CObjectDetection('test')
obj.start(inputVideoName, outputVideoName)
 
    
 
