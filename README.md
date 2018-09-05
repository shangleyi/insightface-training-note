# Insightface Training Note
For two months, I have been training face recognition model using deepinsight’s open source project insightface.
All my experiments were conducted on the Tesla P40 GPU.

## 1. Selecting the network and loss function
Insightface provides a variety of network and loss function choices, but according to the author, training ArcFace with LresNet100E-IR yields the most accurate model which achieved LFW 99.8%+ and MegaFace 98.0%+. Unfortunately, the training process was too slow so I jumped to the second best option – training LrestNet50E-IR with CosineFace. I used the given cleaned ms1m dataset as my training set and after about 400000 iterations the most accurate model I got yields lfw 99.7% and agedb_30 97.6%.

## 2. Saving the logging, standard output and model
I set the parameter ckpt to 2 in order to save all models (else only models that achieved lfw 99%+ will be saved). In order to know which saved model is the best one in the quickest way, I saved the logging and redirected the standard output to be appended to the log.

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110632.png)  
