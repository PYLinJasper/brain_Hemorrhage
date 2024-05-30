'''
Python version 3.8.18
Softmax input
'''

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input, Concatenate
from keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, hamming_loss

from input_config import data_generate, get_args, create_folder

# Function to create a replicated input tensor
def replicate_channels(input_tensor):
    ''' Replicate the single channel to create a 3-channel input'''
    replicated_input = Concatenate()([input_tensor] * 3)
    return replicated_input

def xception_builder(img_size, img_label):
    '''build a Xception model'''
    # Define the input shape
    input_shape = (img_size, img_size, 1)

    # Number of output labels (adjust as needed)
    num_labels = len(img_label)

    # Create an Input layer
    input_layer = Input(shape=input_shape)

    # Replicate the single channel to create a 3-channel input
    replicated_input = replicate_channels(input_layer)

    # Base Xception model (without top layers)
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3),  # Use 3 channels
        input_tensor=replicated_input,
        pooling='avg'
    )

    # Freeze the layers of the base model
    base_model.trainable = False

    # Custom top layers
    flatten_layer = layers.Flatten()(base_model.output)
    dense_layer = layers.Dense(512, activation='relu')(flatten_layer)
    dropout_layer = layers.Dropout(0.5)(dense_layer)
    output_layer = layers.Dense(num_labels, activation='softmax')(dropout_layer)

    # Create the final model
    xception_model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile the model
    xception_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Display the model summary
    xception_model.summary()

    return xception_model

def scheduler(epoch):
    '''learning rate scheduler'''
    if epoch > 10:
        lr = 1e-4

    else:
        lr = 1e-3

    return lr

def print_history(history):
    '''print out model training history'''
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()

def pred(probabilities):
    # Get the index with the maximum probability for each sample
    max_indices = np.argmax(probabilities, axis=1)
    
    # Create a one-hot encoded matrix
    num_classes = probabilities.shape[1]
    one_hot_predictions = np.zeros_like(probabilities)
    
    # Set the corresponding index to 1 in each row
    one_hot_predictions[np.arange(len(one_hot_predictions)), max_indices] = 1
    
    return one_hot_predictions

def model_eval(y_test, y_pred, img_label):
    '''matrix for model evaluation'''
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    h_loss = hamming_loss(y_test, y_pred)
    print(f'Hamming Loss: {h_loss * 100:.2f}%')

    metrics = precision_recall_fscore_support(y_test, y_pred)
    pre_list = []
    re_list = []
    for pre, re, f1, su in zip(metrics[0], metrics[1], metrics[2], metrics[3]):
        pre_list.append(f'{pre * 100:.2f}%')
        re_list.append(f'{re * 100:.2f}%')

    matrics_table = pd.DataFrame()
    matrics_table['Class'] = img_label
    matrics_table['Precision'] = pre_list
    matrics_table['Recall'] = re_list
    matrics_table = matrics_table.set_index('Class')
    print(matrics_table)

    conf_mx = confusion_matrix(y_test, y_pred)

    # Remove diagonal for better viewing
    row_sum = conf_mx.sum(axis=1, keepdims=True)
    nconf_mx = conf_mx/row_sum
    # np.fill_diagonal(nconf_mx,0)

    labels = img_label

    sns.heatmap(nconf_mx, xticklabels=labels, yticklabels=labels, cmap="summer", annot=True)

def train_model(img_size, img_label, x_train, y_train, x_val,y_val, model_path):
    # model build
    xception_model = xception_builder(img_size, img_label)

    # model training
    changeLr = LearningRateScheduler(scheduler)

    self_callback = [
        EarlyStopping(monitor = "loss", patience = 20, verbose = 1, mode = "auto",),
        # ModelCheckpoint(modelPath, verbose = 0, save_best_only = False, save_weights_only = True),
        changeLr
    ]
    epochs = 5

    history = xception_model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=64,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks = self_callback)

    # print out training history
    print_history(history)

    # save model
    xception_model.save(model_path)

def main(args):
    '''main function'''

    rpath = args.root_path
    img_size = args.size
    multi_class = True

    # Generate input data from raw
    # including contour and image augmentation
    x_train, y_train, x_test, y_test, x_val, y_val, img_label = data_generate(rpath,
                                                                        img_size, multi_class)

    # print(x_train.shape)
    weight_path = './weights'
    create_folder(weight_path)
    model_name = args.model + '.h5'
    model_path = weight_path + '/' + model_name

    # train model
    if args.pred:
        train_model(img_size, img_label, x_train, y_train, x_val,y_val, model_path)

    # make prediction
    trained_model = load_model(model_path)
    predict_prob = trained_model.predict(x_test)
    y_pred = pred(predict_prob)

    y_pred_flat = np.argmax(y_pred, axis=1)

    y_test_flat = np.argmax(y_test, axis=1)
    model_eval(y_test_flat, y_pred_flat, img_label)

if __name__ == "__main__":
    ARGS = get_args()
    main(ARGS)
