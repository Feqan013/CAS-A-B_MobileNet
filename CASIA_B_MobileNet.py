
import splitfolders
import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


dest_dirs = {
    'bg': 'D:\\WINDOWS\\CASIA B_3\\bg',
    'cl': 'D:\\WINDOWS\\CASIA B_3\\cl',
    'nm': 'D:\\WINDOWS\\CASIA B_3\\nm'
}
source_path = "D:\\WINDOWS\\GaitDatasetB-silh"


file_lists = {
    'bg': [],
    'cl': [],
    'nm': []
}


for root, dirs, files in os.walk(source_path):
    for file in files:
        for key in file_lists:
            if key in file:
                file_lists[key].append(os.path.join(root, file))
                break


for key, filelist in file_lists.items():
    for name in filelist:
        print(name)
        shutil.copy(name, dest_dirs[key])

input_folder = 'D:\\WINDOWS\\CASIA B_3'
splitfolders.ratio(input_folder, output="D:\\WINDOWS\\CASIA B_3\\Data Folder", seed=1337, ratio=(.70, .30), group_prefix=None)
print(os.listdir("D:\\WINDOWS\\CASIA B_3\\Data Folder\\"))
SIZE = 256
data_folder_path = "D:\\WINDOWS\\CASIA B_3\\Data Folder"
print(os.listdir(data_folder_path))


train_gen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)
train_flow = train_gen.flow_from_directory(
    os.path.join(data_folder_path, 'train'),
    target_size=(256, 256),
    batch_size=130,
    subset="training"
)
valid_flow = train_gen.flow_from_directory(
    os.path.join(data_folder_path, 'val'),
    target_size=(256, 256),
    batch_size=130,
    subset="validation"
)


base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)

model = Model(base_model.input, x)
opt = RMSprop(
    learning_rate=1e-4,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"
)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_flow,
    epochs=10,
    validation_data=valid_flow
)


plt.figure(figsize=(10, 10))
plt.title("MobileNet")
plt.plot(history.history['accuracy'], label='Training Acc')
plt.plot(history.history['val_accuracy'], label='Testing Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('mobilenet.png', dpi=300, bbox_inches='tight')
plt.show()
