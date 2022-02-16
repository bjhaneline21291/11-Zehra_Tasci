# Pneumonia Detection using Deep Learning with 90% accuracy

During the COVID pandemic, my interest grew towards understanding image data better and applying deep learning concepts.
It helped that there was a free to learn, excellent course launched at the time by [Jovian ML](https://jovian.ml/forum/c/pytorch-zero-to-gans/18).

### 1. Data Source
This dataset hosted on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) contained about 5856 x-ray images with labels as Normal and Pneumonia. 

### 2. Tools
The analysis was performed in Pytorch using Kaggle's free to use resources. The GPU had a weekly limit of 30 hours, which proved sufficient. A thing to watch out is to ensuring to log off GPU whenever it is not needed in order to conserve the usage. Also, more than 1 Kaggle session cannot be open with GPU usage.

Jovian ML was used to commit changes. This is not a necessity for rest of the code to run.

### 3. EDA 
The training data had 5216 x-rays with 3875 cases of pneumonia. The testing data had 624 images with 390 pneumonia cases. The validation data was a very small set of 16 images with 50-50 split of labels. Each image consisted of 3 channels for R,G,B and the pixel size was 224 * 224.

### 4. Modeling Approach

4 Modeling approaches were tried- Logistic Regression, Feed forward Neural Network, Convolution Neural Network and Residual Neural Network.

### 5. Results

The accuracies for each of the 4 models were- 56%, 69%, 74% and 90%.

### 6. Summary

Getting the accuracy bump using a pre trained Resnet showed the power of transfer learning. 
The model accuracy could be given a further boost in a couple of different ways: Training the model on more images and playing around with tuning the hyperparameters further.
