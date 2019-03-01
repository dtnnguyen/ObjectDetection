# Object Detection

This project reads an input video, then uses single shot detection (ssd)
and visual object class data (VOCdevkit) to detects objects in each frame.
A result video will be produced with objects being labeled.

## Finished Video

![Finished Video](https://github.com/dtnnguyen/ObjectDetection/blob/master/ObjectDetection.gif)

# Set up:

This project refers to VOCdevkit and ssd pytorch, thus it needs four subfolders:
data, layers, utils and weights. The following is a guide to set up and
retrieve data for these subfolders.

Navigate to the project folder:

### Download torch file for single shot detection:
	
```
	wget https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
```

### Download fc-reduced VGG-16 Pytorch base network weights:
	
```
	mkdir ./weights
	cd weights
	wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

### Download layers and utils of ssd pytorch:

```	
	mkdir ./layers
	mkdir ./utils
	Navigate to https://github.com/amdegroot/ssd.pytorch
	Download layers/* into your project folder: ./layers/
	Download utils/* into your project folder: ./utils/
```

### Download VOC2012 devkit and uncompress the file into VOCdevkit/:
 
```
	mkdir ./data
	cd data
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
	tar xvzf VOCtrainval_11-May-2012.tar
	rm VOCtrainval_11-May-2012.tar
```

# Reference: 
	
	For more information on Visual Object Class (VOC), see this link:
	https://pjreddie.com/media/files/VOC2007_doc.pdf 


