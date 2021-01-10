import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Function Definitions
#region

def create_result_container_directory(path, directory_name):

    if os.path.exists(directory_name):
        pass
    else:
        new_directory = path + "\\" + directory_name

        try:
            os.mkdir(new_directory)
        except OSError:
            print ("Creation of the directory failed")
        else:
            print ("Successfully created the directory")

def load_model_json(directory):

    json_file = open(str(directory + ".json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load model parameters into loaded model
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    return loaded_model.load_weights(str(directory + ".h5"))

def remove_substring_from_directory(substring, replacement_string, path):
    converted = False
    exists = True
    for folder in os.listdir(path):
        if converted == False:
            if folder.find(substring) > -1:
                s = folder.split(" ")
                os.rename(os.path.join(path, folder), os.path.join(path, s[0]))
                converted = True
        else:
            pass

def create_accuracy_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')

def create_loss_graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')

def save_model(model):
    # This removes the "(latest)" substring from the end of the last complete cycle so that a new one can be place on this time
    remove_substring_from_directory("(latest)", "", str(os.getcwd() + "\\Results"))

    # Directory setup to make the storing of multiple results easier to navigate
    result = str(round(scores[1]*100, 2))
    os.mkdir(str(results_directory_name + "/" + result + " (latest)"))
    directory_path = str(results_directory_name + "/" + result + " (latest)" + "/")
    file_name = str(result + ".json")
    final_path = directory_path + file_name

    # Save a .json file and a .h5 file
    model_json = model.to_json()
    with open(final_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(str(final_path + ".h5"))
    plt.savefig(str(final_path) + ".png")

#endregion

# Architectures
#region



#endregion


results_directory_name = "Results"
create_result_container_directory(os.getcwd(), results_directory_name)
directory = r'C:\Users\jonny\Desktop\dataset\dataset\train'

image_width = 48
image_height = 48
image_channels = 1
color_mode = 'grayscale'
class_mode = 'categorical'
batch_size = 16
seed = 101


data_generator_train = ImageDataGenerator(rescale=1./255, validation_split=0.1, dtype=tf.float32)     # create a validation set that consists of 20% of the images in the training set
data_generator_test = ImageDataGenerator(rescale=1./255)

train_data_gen = data_generator_train.flow_from_directory(
    directory = directory,  # path to the directory containing the subclasses
    target_size = (image_width, image_height),    # the size of the images will be set to this
    color_mode = color_mode,     # determine whether the image is grayscale, rgb or rgba
    class_mode = class_mode,     # there are more than 2 classes, so this is set to categorical
    shuffle = True,      # shuffle the images
    seed = seed,
    batch_size = batch_size,  # the batch size must be a number that is divisble by the total number in the file
    subset = 'training')      # this is the training portion of the data

validation_data_gen = data_generator_train.flow_from_directory(
    directory = directory,      # path to the directory containing the subclasses
    target_size = (image_width, image_height),    # the size of the images will be set to this
    color_mode = color_mode,     # determine whether the image is grayscale, rgb or rgba
    class_mode = class_mode,     # there are more than 2 classes, so this is set to categorical
    shuffle = True,       # shuffle the images
    seed = seed,
    batch_size = batch_size,  # the batch size must be a number that is divisble by the total number in the file (1 is set but will be slower)
    subset = 'validation')    # this is the validiation portion of the data

# Set the number of trainig and validation steps by accessing the total number of samples for each and divide by the batch size
train_steps = train_data_gen.samples/batch_size
val_steps = validation_data_gen.samples/batch_size

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=[image_width, image_height, image_channels]),
        keras.layers.Dense(300, activation = 'swish'),
        keras.layers.Dense(100, activation = 'swish'),
        keras.layers.Dense(7, activation = 'softmax')
    ]
)

model.compile(loss=[keras.losses.CategoricalCrossentropy(from_logits=True)], optimizer=keras.optimizers.SGD(lr=0.01), metrics=["accuracy"])

history = model.fit(train_data_gen, epochs=40, steps_per_epoch=train_steps, validation_data=validation_data_gen, validation_steps=val_steps)

scores = model.evaluate(validation_data_gen, verbose=0)
percentile_score = round(scores[1]*100, 2)
print("%s: %f" % (model.metrics_names[1], percentile_score))

# Graph Results
create_accuracy_graph(history)
#create_loss_graph(history)

# Save Model & Figure
save_model(model)
