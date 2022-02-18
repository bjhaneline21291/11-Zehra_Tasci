# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

print(os.getcwd())




# initializing learning rates and bath size for training
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


print("loading images")

#loading dataset images from specific folder
dataset_path=os.getcwd()+"//dataset"

imagePaths = list(paths.list_images(dataset_path))

imagePaths = [imagePath.replace("\\","//",-1) for imagePath in imagePaths]
#for storing images and their labels in data format to work with it easily
data = []
labels = []


for imagePath in imagePaths:
	#file name
	label = imagePath.split("//")[-2]


	image = load_img(imagePath, target_size=(224, 224))

	image = img_to_array(image)

	image = preprocess_input(image)


	data.append(image)
	labels.append(label)

#commands for numpy arrays conversion
data = np.array(data, dtype="float32")

labels = np.array(labels)

#commands for converting labels to binary format
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#partioning of dat into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, stratify=labels, random_state=42)

#different snaps of images with different properties

aug = ImageDataGenerator(zoom_range=0.14,width_shift_range=0.3,rotation_range=22,shear_range=0.15,horizontal_flip=True,fill_mode="nearest",height_shift_range=0.3)

# loading base model from mobile net

baseModel = MobileNetV2( include_top=False,weights="imagenet",input_tensor=Input(shape=(224, 224, 3)))

# contructing head model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# placing head model on top of base model

model = Model(inputs=baseModel.input, outputs=headModel)

# avoid training of base model durring training
for layer in baseModel.layers:
	layer.trainable = False



print("compiling model")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


print("training head")

H = model.fit(aug.flow(trainX, trainY, batch_size=BS),steps_per_epoch=len(trainX) // BS,validation_data=(testX, testY),validation_steps=len(testX) // BS,epochs=EPOCHS)


print("Checkin Model")

predIdxs = model.predict(testX, batch_size=BS)


predIdxs = np.argmax(predIdxs, axis=1)


print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))


print("saving model")
model_path=os.getcwd()+"//model//mask_model"
model.save(model_path+".h5")

# plotting training loss and training accuracy

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch ")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plot_path=os.getcwd()+"//plot//plot.png"
plt.savefig(plot_path)
