# Camera Calibration in C++

C++ Implementation of Zhang's camera calibration method.  
Based on : 
[A Flexible New Technique for Camera
Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) and [Zhang's Camera Calibration Algorithm: In-Depth Tutorial and Implementation](https://www.researchgate.net/publication/303233579_Zhang's_Camera_Calibration_Algorithm_In-Depth_Tutorial_and_Implementation) papers.

*Note that this implementation is inferior to what you can find in most popular CV libraries. I wrote it to learn how the whole process of calibrating a camera looks like and how you can find a camera pose just from n pairs of known 3D points(on a planar surface) and relating sensor points.

## Dependencies

* C++  
* OpenCv 3.4
* Eigen 3.4.0