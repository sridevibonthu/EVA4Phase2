
# Transfer Learning on Custom Dataset using MobileNet V2

## Table of Contents

* [Problem](#Problem-Statement)
* [Requirements](#Prerequisites-and-Requirements)
* [My Team and Mentor](#My-Team-and-Mentor)
* [Data](#Dataset)
* [Resizing Strategy](#Resizing-Strategy-Followed)
* [Model](#The-Model-I-trained)
* [Graphs](#Accuracy-and-Loss-Curves)
* [Misclassified Images for each of the classes](#Misclassified-Images)
* [Deploying and testing](#Deploying-on-AWS)

Note : The Resizing Strategy and Model sections explain the code

## Problem Statement

A custom dataset on flying objects need to be curated. Each batch is asked to collect 1000 images each. We have to tain that custom datset of 4 classes using MobileNet V2 through transfer learning. the trained model need to be uploaded on Lambda for future use.

## Prerequisites and Requirements

### prerquisites 

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [AWS Account](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc)
* [Serverless](https://www.serverless.com/) 
* [Insomnia](https://insomnia.rest/download/)
* [Google Colab](https://colab.research.google.com/)
* [python-resize-image](https://pypi.org/project/python-resize-image/)

### requirements [here]

## My Team and Mentor

### My Team

  - [Sridevi B](https://github.com/sridevibonthu) , [Sridevi_on_LinkedIn](https://www.linkedin.com/in/sridevi-bonthu/)
  - [Anilkumar N Bhatt](https://github.com/anilbhatt1) , [Anil_on_LinkedIn](https://www.linkedin.com/in/anilkumar-n-bhatt/)
  - [Gajanana Ganjigatti](https://github.com/gaju27) , [Gaju_on_LinkedIn](https://www.linkedin.com/in/gajanana-ganjigatti/)
  - [Maruthi Srinivas](https://github.com/mmaruthi) , [Maruthi_on_LinkedIn](https://www.linkedin.com/in/maruthi-srinivas-m/)
  - [SMAG TEAM](https://github.com/SMAGEVA4/session1/tree/master/Session1) :performing_arts: team github account

### My Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) , [The School of A.I.](https://theschoolof.ai/)

## Datset

- Took the images from google drive shared location where images are collected by crowd-souring. 
- Total 19318 images belonging to 4 flying objects were collected.
- These images were located in 4 folders 'Flying Birds', 'Large QuadCopters', 'Small QuadCopters' and 'Winged Drones'. 
- Images from all these folders were brought into a zip file named 'ThumbnailData.zip' and saved to personal gdrive folder. 

## Resizing Strategy Followed

- Data is collected by many people and from many sources. All the images are of different resolution and of different extensions.
- I have taken only jpg files. the code followoed for flyind birds folder is ...
```
flyingbirdspath = os.path.join('/content/drive/My Drive/EVA4/S2Data', "Flying Birds")
flyingbirdslist = list(filter(lambda x: x.endswith('jpg'), os.listdir(flyingbirdspath)))

```
- I need to follow a best strategy to resize images, as resizing them in transformations slows down the training process.
- Challenges : I can use Thumbnail method of Pillow, but its maximum resolution is 128. I can write a code to preserve aspect ratios by considering width and heights of the current image, but in this approach i am losing part of the picture
- python-resize-image package helped to me to resize the images to 224X224 by retaining the aspect ratio. Created a zip file with folders for every class and placed the resized images in that zip file with the help of the following code segment.
```
from resizeimage import resizeimage
def writeToZip(impath, imlist, imtype, zipfolder):
  for i in tqdm(range(len(imlist))):
    img = Image.open(os.path.join(impath, imlist[i])).convert('RGB')
    img = resizeimage.resize_thumbnail(img, [224, 224])
    img.save("resized.jpg", img.format)
    zipfolder.write('resized.jpg', f'Data/{imtype}/img{i:0>5d}.jpg')

```
- The source code for resizing the data and placing it into a zip file is [here](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/Phase2Session2_zip_file_creation.ipynb).

## The Model I trained

- Torch version was downgraded to torch==1.5.1+cu92 and torchvision==0.6.1+cu92 because AWS lambda was not able to load model against latest 1.6.0 versions due to space constraints.
- Mobilenet-V2 with 3,504,872 parameters and convolution blocks were named as features and Dense Layers as classifier.
- Pattern of Conv2d -> BN -> Relu6 -> Conv2d -> BN -> Relu6 -> Conv2d -> BN was used in each layers. 
- I have changed the last classifier section to suit number of classes in our problem. 
```
inputs = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Linear(inputs, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(256, num_classes))
```
- It is observed that the test accuracy is more than train accuracy and also identified that few more layers need to be unfrozen as the objects were not present in Imagenet dataset.
- the 17th and 18th layers were unfreezed in the following way.
```
for name, child in model.features.named_children():
  if name in ['17', '18']:
    print("Layer ", name , " of features stack is unforzen")
    for param in child.parameters():
      param.requires_grad = True
```
- The number of trainable and non-trainable parameters after this change are......

  ![parameters](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/change%20in%20model.png)
  
- The Model was trained for 25 epochs and achieved a test accuracy of 87.85%. [Logs are available in the source code](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/EVAP2S2_Assignment2_V4_torchdowngrade.ipynb)
- OneCyclePolicy is used with max_lr of 0.01 and optimizer adopted was SGD
- Took care to save the jit tracecd model by moving it to cpu.


## Accuracy and Loss Curves

| Train and Test Accuracy Vs Epochs| Train and Test Loss Vs Epochs|
| ------------- | ------------- |
| ![Accuracy](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/accuracy%20curve.png)| ![Loss](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/loss%20curves.png) |

## Misclassified Images

| Class  | Gallery of Misclassified images |
| ------------- | ------------- |
| Large Quadcopters  | ![Misclassified Large Quadcopters](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/misclassified_Large_Qcopters.png)  |
| Small Quadcopters  | ![Misclassified Small Quadcopters](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/misclassified_Small_Qcopters.png)  |
| Winged Drones  | ![Misclassified Winged Drones](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/misclassified_WingedDrones.png)  |
| Flying Birds  | ![Misclassfied Flying Birds](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/misclassified_birds.png)  |



## Deploying on AWS

- The jit traced trained model is saved and stored in a S3 Bucket.
- the model is deployed on AWS Lambda and tested using Insomnia

| Input Image | Prediction through Insomnia |
| ------------- | ------------- |
| ![Winged Drone](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/best-fixed-wing-drones.jpg) | ![prediction](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/prediction_wingeddrone.PNG) |
| ![Flying Birds](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/FlyingBird.jpg) | ![Prediction](https://github.com/sridevibonthu/EVA4Phase2/blob/master/Session2/prediction_flyingbirds.PNG) |



