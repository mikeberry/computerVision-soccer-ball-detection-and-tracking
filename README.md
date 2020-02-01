# computerVision-soccer-ball-detection-and-tracking

A basic implementation of a detecter (YOLOv3) and tracker (CORS) to track a socker ball on a field.
The implementation is so that each 10 frames the detecter is looking for the ball. The rest of the
frames the ball gets tracked by the tracker. In case the detector is not able to find the ball e.g.
because of occlusion, the tracker keeps responsibility.

This project is part of the official openCV computer Vision I course. 
It had do be solved without a sample solution.  
https://opencv.org/courses/