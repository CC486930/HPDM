# High-precision detection modeling reveals fundamental structural scales of amorphous materials
## Overview
Here we give the representative code for the High-precision detection model (HPDM) and the representative dataset generated for the results in the main text (in dataset). The specific execution process is as follows:

## Requirements
 * Python = 3.7
 * TensorFlow  = 2.x
 * Torch = 1.11
 * Torchvision = 0.4.1
 * Numpy = 1.x
 * 
## Usage
First, collect training and validation data and deploy it like this(for binary classification),
```
positives/
  1zzcons_0512_-1.0_0.840_unstable_201.jpg
  1zzcons_0512_-1.0_0.840_unstable_202.jpg
  ...
negatives/
  1zzcons_0512_-1.0_0.832_stable_0.jpg
  1zzcons_0512_-1.0_0.832_stable_1.jpg
  ...
positives_t/
  1zzcons_0512_-1.0_0.840_unstable_22.jpg
  1zzcons_0512_-1.0_0.840_unstable_35.jpg
  ...
negatives_t/
  1zzcons_0512_-1.0_0.834_stable_0.jpg
  1zzcons_0512_-1.0_0.834_stable_1.jpg
  ...
```
and then run the scripts in sequence.
```
python convert.py
```
  Convert image resolution.  
```
python preprocess.py
```
  Data preprocessing, splitting the amorphous image into blocks of “box” parameter size and processing the data into standard lib data.<br>  
```
python spore.py
```
  Train, after training, you will get the best model "checkpoint_50_best.pth". (after testing, the size of the batchsize does not affect the inference time, so 128 is chosen.)  
```
python probmap.py
```
  Use the trained model to return the probability matrix of the whole image.<br>  
```
python heatmap_ori.py
```
  Generate structured segmented images of test data.<br>  
```
python report.py
```
  Generate categorized information about the test data.<br>
