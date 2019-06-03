# statement:
**you should obey the following rules:**  
1. This database will only be used for research purposes.   
2. I will not make any part of this database available to a third party.   
3. I'll not sell any part of this database or make any profit from its use.  
# Julyedu handwriting recognition competition
the competition link is:http://jingsai.julyedu.com/v/25820185621436242/detail.jhtml  
you can download the data files:train_july.csv,test_july.csv,sample_submission.csv  
from above webpage, or from my baidu pan link:  
https://pan.baidu.com/s/1Hj_ChsGrQfFdlqzV-cdqTA   
Extraction code:g8t5 

# Required packages
pandas  
pytorch  
torchvision  
tqdm  
signal  

# Document description
**mnist_models.py** build up some small network models  
**main.py** train your model and save it, at the same time it predict the result using  
test_july.csv and will generate a submission.csv that is like sample_submission.csv in
in structure and format.  
**debug.py** is used for debugging.In my process, I used it to load my generated model
to test whether it can prediect the result correctly or not.  
**models** document a subfile "MNIST" in which a my pre-trained model is.  
