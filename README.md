# Structure_From_Motion_ArUco_Marker_3D_Reconstruction


Structure from Motion from Two Views Structure from Motion is the process of estimating the 3D structure of a scene from a set of 2D images. 
Therefore, we need to estimate the poses of a calibrated camera from two images, then reconstruct the 3D structure of the scene up to an known scale factor. 
The actual scale factor is recovered by detecting an object of a known size (here we use ArUco Marker to detect the distance of two markers). 
The algorithm consists of the following steps:

Find camera parameters
Find corresponding points
Estimate fundamental and essential matrix
Detect Aruco Marker and calculate the distance.
Recover the actual scale, resulting in a metric reconstruction
Determine the 3D location of the matches points using triangulation
The code we implemented takes in two 2D pictures of an object from different angles as input and it outputs a 3D point cloud that captures the structure of the original object. 
In this project we consider camera calibration and extrinsic properties such as optical center, focal length etc. We first calibrate the camera using a checkerboard to get the intrinsic parameters. 
In order to calculate the fundamental matrix for transformation from image A to image B we calculates t In this project, we work with the SfM algorithm. Our algorithm further uses markers to validate the distance calculation from multiple views. 
We chose Aruco Marker because they are often claimed to be useful in 3D reconstruction. These markers are highly detectable and works well in identifying features that 3D reconstruction can use to overcome challenging scene characteristics such as repetitive patterns
