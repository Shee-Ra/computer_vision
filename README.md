# Computer vision

Report and code I did for a computer vision course at Melbourne University. There are six main folders, each containing a key milestone for the course.

The culmination of this course was making an animation of depth maps, which can be seen here: [https://youtu.be/HwIKfbiJYAU](https://youtu.be/HwIKfbiJYAU)

## Final report
The final report is in the folder 4_Report

## Compulsory work
Compulsory work is covered in these folders:
* 1_1D_spatial_cross_correlation
* 2_2D_spatial_cross_correlation
* 3_Using_FFT_for_image_signal_comparison
* 5_Depth_maps

Each of the above have:
* An associated worksheet in docs
* Code which performs the tasks outlined in the worksheet is a script in the folder
* Supporting code (functions etc.) is found in the folder `src/`
* Provided data is in the folder `data/`
* Results are in the form of graphs output to `data/results`
* Notebooks used for exploration are in the folder `notebooks/`

To run scripts, use the command line. For example to calculate the 1D cross-correlation of the sensor data, assuming you are in the `computer_vision` folder type:

```
python 1_1D_spatial_cross_correlation/script_to_get_offset.py
```

Where there exists performance tests, they are done in the command line. See the files called `performance_tests.py` in the relevant folder for more information on how to run them.

## Extension project
The extension project involved using the code from `5_Depth_maps` and using it to create a depth map of right/left stereo image pairs from the [DrivingStereo](https://drivingstereo-dataset.github.io/) project.

Data used to create depth maps is found here:[DrivingStereo](https://drivingstereo-dataset.github.io/) > testing data > Left (Right) images Download > left(right)-image-half-size > 2018-08-01-11-13-14.zip

To get depth maps run: `1_get_depth_maps.py`