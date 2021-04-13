import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from .data_preprocessing import get_dataset, data_augmentation

# loading the dataset
images, labels = get_dataset()

# splitting the data
train_images, test_images, train_labels, test_labels = train_test_split(images, labels,
                                                                        test_size=0.4, stratify=labels)
valid_images, test_images, valid_labels, test_labels = train_test_split(test_images, test_labels,
                                                                        test_size=0.5, stratify=test_labels)

# adding one extra dimension to each data collection
train_images = np.expand_dims(train_images, -1)
valid_images = np.expand_dims(valid_images, -1)
test_images = np.expand_dims(test_images, -1)

# perform augmentation
train_data = data_augmentation(train_images, train_labels)

# BUILDING THE CNN
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = len(set(train_labels))
EPOCHS = 200
LEARNING_RATE = 0.003
OPTIMIZER = Nadam(LEARNING_RATE)

CALLBACKS = [ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, mode='max', patience=20, verbose=1),
             EarlyStopping(monitor='val_accuracy', mode='max', patience=25, verbose=1,
                           restore_best_weights=True),
             ModelCheckpoint('model.h5', monitor='val_accuracy', mode='max', save_best_only=True)]

input_layer = Input(INPUT_SHAPE)

# 1st convolutional block
conv = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(input_layer)
batchnorm = BatchNormalization()(conv)
conv = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(batchnorm)
batchnorm = BatchNormalization()(conv)
pooling = MaxPooling2D()(batchnorm)

# 2nd convolutional block
conv = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(pooling)
batchnorm = BatchNormalization()(conv)
conv = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(batchnorm)
batchnorm = BatchNormalization()(conv)
pooling = MaxPooling2D()(batchnorm)

dropout = Dropout(0.3)(pooling)

# 3rd convolutional block
conv = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(dropout)
batchnorm = BatchNormalization()(conv)
conv = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(batchnorm)
batchnorm = BatchNormalization()(conv)
pooling = MaxPooling2D()(batchnorm)

# 4th convolutional block
conv = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(pooling)
batchnorm = BatchNormalization()(conv)
conv = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(batchnorm)
batchnorm = BatchNormalization()(conv)
pooling = MaxPooling2D()(batchnorm)

avg = GlobalAveragePooling2D()(pooling)

output_layer = Dense(units=NUM_CLASSES, activation='softmax')(avg)
model = Model(input_layer, output_layer)

model.compile(optimizer = OPTIMIZER,
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(train_data,
        epochs = EPOCHS,
        validation_data = (valid_images, valid_labels),
        callbacks = CALLBACKS)