

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import visualkeras
import os
for dirname, _, filenames in os.walk('../Sign Language dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')

# %%
train_df = pd.read_csv('../Sign Language dataset/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('../Sign Language dataset/sign_mnist_test/sign_mnist_test.csv')

# %%
print(len(train_df))
print(len(test_df))

# %%
train_df.head()

# %%
test_df.head()

# %%
train_df.info()

# %%
test_df.info()

# %%
train_df.isnull().sum()

# %%
sns.heatmap(train_df.isnull())

# %%
sns.heatmap(train_df.corr())

# %%
test_df.isnull().sum()

# %%
sns.heatmap(test_df.isnull())

# %%
sns.heatmap(test_df.corr())

# %%
train_df.shape,test_df.shape

# %%
plt.figure(figsize = (10,8)) 
sns.set_style("darkgrid")
sns.countplot(data = train_df, x = train_df['label'])

# %%
y_train = train_df['label']
y_test = test_df['label']

x_train = train_df.drop('label',axis=1)
x_train = np.array(x_train,dtype='float32')/255

x_test = test_df.drop('label',axis=1)
x_test = np.array(x_test,dtype='float32')/255

print('x_train : {}\n'.format(x_train[:]))
print('Y-train shape: {}\n'.format(y_train))
print('x_test shape: {}'.format(x_test.shape))

# %%
x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train, test_size=.05, random_state=1234,)

# %%
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# %%
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# %%
plt.figure(figsize=(20, 8))
x, y = 10, 4 
for i in range(40):  
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape(28,28), cmap='gray_r')
plt.show()

# %%
datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range = 0.2,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(x_train)

# %%
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1, factor=0.5, min_lr=0.00001)

# %%
model = Sequential()

# %%
model.add(Conv2D(45, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), strides = 1, padding = 'same'))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides = 2, padding = 'same'))

model.add(Conv2D(55, kernel_size=(3,3), strides=1, padding='same', activation='relu'))

model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=24,activation='softmax'))
model.summary()


# %%
visualkeras.layered_view(model, legend = True)

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
hist= model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=10, validation_data = (x_test,y_test), callbacks = [learning_rate_reduction])

# %%
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

# %%
epochs = [i for i in range(10)]

fig,ax = plt.subplots(1,2)
train_acc = hist.history['accuracy']
train_loss = hist.history['loss']
vall_acc= hist.history['val_accuracy']
vall_loss=hist.history['val_loss']
fig.set_size_inches(10,6)

ax[0].plot(epochs, train_acc, color = 'red', marker = 'o', linestyle = '-', label='Train Acc')
ax[0].plot(epochs, vall_acc, color = 'blue', marker = 'o', linestyle = '--', label= 'Test Acc')
ax[0].set_title('Train and Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(epochs, train_loss, color = 'red', marker = 'o', linestyle = '-', label='Train Loss')
ax[1].plot(epochs, vall_loss, color = 'blue', marker = 'o', linestyle = '--', label= 'Test Loss')
ax[1].set_title('Train and Validation Loss')
ax[1].legend()
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')

plt.show()


