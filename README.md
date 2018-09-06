# InsightFace Training Note
For two months, I have been training face recognition model using Deep Insight’s open source project InsightFace (https://github.com/deepinsight/insightface.git).
All my experiments were conducted on the Tesla P40 GPU.

## 1. Selecting the network and loss function
InsightFace provides a variety of network and loss function choices, but according to the author, training ArcFace with LresNet100E-IR yields the most accurate model which achieved LFW 99.8%+ and MegaFace 98.0%+. Unfortunately, the training process was too slow so I jumped to the second best option – training LrestNet50E-IR with CosineFace. I used the given cleaned ms1m dataset as my training set and after about 400000 iterations the most accurate model I got yields lfw 99.7% and agedb_30 97.6%.

## 2. Saving the logging, standard output and model
I set the parameter ckpt to 2 in order to save all models (else only models that achieved lfw 99%+ will be saved). In order to know which saved model is the best one in the quickest way, I saved the logging and redirected the standard output to be appended to the log.

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110632.png)
![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110723.png)

## 3. Creating private training dataset

### Alignment
To detect, align and crop the faces, use /src/align/align_megaface.py with argument "--name webface". However the codes are for photos which have attributes bbox and landmark, so I added the following codes:

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904105710.png)

Now landmark is a 10x1 numpy array, containing x and y coordinates of 5 facial landmarks. But only 3 points are needed for estimation, so I changed the line after similarity transform to

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180904110506.png)

### Change file format and rename
After running align_megaface.py I got a dataset containing only cropped faces. It also generates a .lst file which can be directly used for the next step.
However, if names of photos contain Chinese elements, there will be an exception during training. Also, the default raw photo format for the training is .jpg. To solve these two problems, I wrote a script which uses PIL to convert all photos to RGB, rename and save them as .jpg files.

### Generate .lst file
I modified glint2lst.py so that it writes all photo names to a .lst file in the format 1 ADDRESS LABEL.

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180905151902.png)

Notice that the folders' structure should look like the following:
>dataset
>>folders of different label each represents an individual
>>>photos of one individual

### Write the property file and use face2rec2.py to generate .idx and .rec
The property file has the format <TOTAL NUMBER OF IDENTITIES,112,112>. The codes in /src/face2rec2.py directly generate a new property file when merging two datasets.

### Merge the private dataset with ms1m
I used /src/data/dataset_merge.py to merge the two datasets.

## 4. Using triplet to fine-tune
Using train_triplet.py to fine-tune the model sometimes can improve the accuracy by about 0.1%. All the parameters are given in insightface’s readme page.

## 5. Verifying accuracy
I used /src/eval/verification.py to verify the accuracy of my model.

## Result (verification datasets are from insightface)
|                        | LFW(%)  | CFP-FF(%)  | CFP-FP(%)  | AgeDB-30(%)  |
| ----------------       | ------  | ---------  | ---------  | -----------  |
| R50 (CosineFace)       | 99.717  | 99.814     | 92.714     | 97.600       |
| R50 (triplet)          | 99.717  | 99.800     | 93.114     | 97.783       |
| MobileFaceNet(ArcFace) | 99.483  | 99.429     | 90.043     | 95.550       |
| MobileFaceNet(triplet) | 99.650  | 99.657     | 90.529     | 96.317       |

![](https://github.com/shangleyi/insightface-training-note/raw/master/QQ截图20180906082939.png)

## MobileFaceNet training process
The first two steps are from: https://github.com/deepinsight/insightface/issues/214. The dataset I used combined the ms1m-v1 dataset from InsightFace and a private dataset. The private dataset is provided by Shenzhen Sunwin Intelligent and contains 1,900,000 raw photos of 50,000 identities collected from Chinese social media Weibo, QQ, Wechat, Tik Tok, etc. No data overlap with ms1m, lfw and agedb-30 is detected yet. The merged dataset contains around 135k individuals.

After 140k iteration the highest accuracy on agedb-30 is 89.333%. I used the 89.3% model as the pretrained model and trained with argument "--lr_steps='100000,140000,160000'". After 400k iteration the highest accuracy on agedb-30 is 94.817%. I used the 94.8% model as the pretrained model and trained it on ms1m-v1 from InsightFace. After 600k iteration the highest accuracy on agedb-30 is 95.767%. I then fine-tuned the 95.7% model using /src/train_triplet.py and after 30k iteration I got the above agedb-30 96.317% result.

## Citation
```
@article{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Zafeiriou, Stefanos},
journal={arXiv:1801.07698},
year={2018}
}
```

## Contact
Leyi Shang(leshang@ucsd.edu / shangleyi@outlook.com)
