# Insightface Training Note
For two months, I have been training face recognition model using deepinsight’s open source project insightface.
All my experiments were conducted on the Tesla P40 GPU.

## 1. Selecting the network and loss function
Insightface provides a variety of network and loss function choices, but according to the author, training ArcFace with LresNet100E-IR yields the most accurate model which achieved LFW 99.8%+ and MegaFace 98.0%+. Unfortunately, the training process was too slow so I jumped to the second best option – training LrestNet50E-IR with CosineFace. I used the given cleaned ms1m dataset as my training set and after about 400000 iterations the most accurate model I got yields lfw 99.7% and agedb_30 97.6%.

## 2. Saving the logging, standard output and model
I set the parameter ckpt to 2 in order to save all models (else only models that achieved lfw 99%+ will be saved). In order to know which saved model is the best one in the quickest way, I saved the logging and redirected the standard output to be appended to the log.

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110632.png)
![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110723.png)

## 3. Creating private training dataset
I was given 190,000 raw photos of 50,000 identities. The faces must be cropped to 112x112 and aligned using /src/align/align_megaface. For the argument “name” I used webface.

### Alignment
Codes in align_megaface.py are for photos which have attributes bbox and landmark, but mine did not. So I added the following codes:

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904105710.png)

Now landmark is a 10x1 numpy array, containing x and y coordinates of 5 facial landmarks. But only 3 points are needed for estimation, so I changed the line after similarity transform to

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110506.png)

### Change file format and rename
After running align_megaface.py I got a dataset containing only cropped faces. It also generates a .lst file which can be directly used for the next step.
However, if names of photos contain Chinese elements, there will be an exception during training. Also, the default raw photo format for the training is .jpg. To solve these two problems, I wrote a script which uses PIL to convert all photos to RGB, rename and save them as .jpg files.

### Generate .lst file
I modified glint2lst.py so that it writes all photo names to a .lst file in the format 1 ADDRESS LABEL.

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180905151902.png)

Notice that the folders' structure should strictly follow the rule:
>dataset
>>folders of different label each represents an individual
>>>photos of one individual

## 4. Using triplet to fine-tune
Using train_triplet.py to fine-tune the model can improve the accuracy by about 0.1%. All the parameters are given in insightface’s instruction.

## 5. Verifying accuracy
Use verification.py in /src/eval/ to verify accuracy.

## Result (verification datasets are from insightface)
|                           | LFW(%)  | CFP-FF(%)  | CFP-FP(%)  | AgeDB-30(%)  |
| ----------------          | ------  | ---------  | ---------  | -----------  |
| R50 (CosineFace)          | 99.717  | 99.814     | 92.714     | 97.600       |
| R50 (fine-tune)           | 99.717  | 99.800     | 93.114     | 97.783       |
| MobileFaceNet(ArcFace)    | 99.483  | 99.429     | 90.043     | 95.550       |
| MobileFaceNet(fine-tune)  | 99.433  | 99.457     | 89.614     | 95.733       |
