# High-precision detection modeling reveals fundamental structural scales of amorphous materials
### summarize
Here we give the representative code for the High-precision detection model (HPDM) and the representative dataset generated for the results in the main text (in dataset). The specific execution process is as follows:

* convert.py<br>
  Convert image resolution.<br> 
* preprocess.py<br> 
  Data preprocessing, splitting the amorphous image into blocks of “box” parameter size and processing the data into standard lib data.<br>
* spore.py<br>
  Train, after testing, the size of the batchsize does not affect the inference time, so 128 is chosen.<br>
* probmap.py<br>
  Use the trained model to return the probability matrix of the whole image.<br>
* heatmap_ori.py<br>
  Generate structured segmented images of test data.<br>
* report.py<br>
  Generate categorized information about the test data.<br>
