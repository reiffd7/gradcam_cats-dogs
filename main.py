from pre_processing import *
from clearml import Task
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import albumentations as A
from modeling import *
from pre_processing import *

def plot_precision_recall_accuracy(history, total_training_steps):
    val_precision = history.history['val_precision_m']
    val_recall = history.history['val_recall_m']
    val_accuracy = history.history['val_accuracy']
    plt.figure(figsize=(8,8))
    plt.scatter(total_training_steps, val_precision[len(val_precision)-1], label='Precision')
    plt.scatter(total_training_steps, val_recall[len(val_recall) - 1], label='Recall')
    plt.scatter(total_training_steps, val_accuracy[len(val_accuracy) - 1], label='Accuracy')
    plt.title('Eval')
    plt.xlabel('Iterations')
    plt.legend()
    return plt.show()


if __name__ == '__main__':
    ## Arguments: 
    ##  1 - Batch Size
    ##  2 - Epochs
    ##  3 - Train Path
    ##  4 - Validation Path
    ##  5 - Project Name
    ##  6- Task Name
    BATCH_SIZE = 25
    EPOCHS = 50
    TRAIN_PATH = '../cats-dogs-data/train_baseline_4000'
    VAL_PATH = '../cats-dogs-data/val_1000'
    BASE_MODEL_NAME = 'mobilenetv2'
    PROJECT_NAME = 'caged_cats_model_improvement'

    SAMPLE_SIZE = len(os.listdir(TRAIN_PATH))
    TASK_NAME = BASE_MODEL_NAME + 'sample_size' + str(SAMPLE_SIZE) + 'batch' + str(BATCH_SIZE) + 'epochs' + str(EPOCHS)

    IMG_SIZE = (224, 224, 3)
    TEST_SIZE = 0.2
    RANDOM_STATE = 2018
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 0.0001

    model_name = 'models/{}_epochs{}_batch{}_sample_size{}_.h5'.format(BASE_MODEL_NAME, EPOCHS, BATCH_SIZE, SAMPLE_SIZE)
    class_info = {0: 'Cat', 1: 'Dog'}

    X_train, y_train, train_files = manual_pre_process(TRAIN_PATH, 224, SAMPLE_SIZE)
    X_val, y_val, val_files = manual_pre_process(VAL_PATH, 224, SAMPLE_SIZE)

    training_steps_per_epoch = X_train.shape[0]/BATCH_SIZE
    total_training_steps = EPOCHS * training_steps_per_epoch

    print('Training Steps per Epoch: {}'.format(training_steps_per_epoch))
    print('Total # of Training Steps: {}'.format(total_training_steps))

    # Declare an augmentation pipeline
    # AUGMENTATIONS_TRAIN = A.Compose([
    #     # A.HorizontalFlip(p=0.5),
    #     # A.RandomBrightnessContrast(p=0.2),
    #     A.RandomContrast(limit=0.2, p=0.5),
    #     A.RandomBrightness(limit=0.9, p=0.5),
    #     # A.RandomRotate90(p=0.5),
    #     # A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
    #     #                    val_shift_limit=10, p=.9),
    #     # A.ShiftScaleRotate(p=0.5)
    #     # A.HueSaturationValue(p=0.6)
    # ])

    train_gen = DatasetSequence(X_train, y_train, BATCH_SIZE)
    valid_gen = DatasetSequence(X_val, y_val, BATCH_SIZE)

    input_shape = IMG_SIZE
    x, inp = build_model_top(input_shape)

    mobilenetv2 = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                input_tensor = x,
                                               include_top=False,
                                               weights='imagenet')

    efficient_netb0 = tf.keras.applications.EfficientNetB0(input_shape=input_shape,
                                                    input_tensor = x,
                                                include_top=False,
                                                weights='imagenet')

    efficient_netb1 = tf.keras.applications.EfficientNetB1(input_shape=input_shape,
                                                    input_tensor = x,
                                                include_top=False,
                                                weights='imagenet')

    efficient_netb5 = tf.keras.applications.EfficientNetB5(input_shape=input_shape,
                                                    input_tensor = x,
                                                include_top=False,
                                                weights='imagenet')

    models_dict = {'mobilenetv2': mobilenetv2, 
                'efficientnetb0': efficient_netb0, 
                'efficientnetb1': efficient_netb1,
                # 'efficientnetb2': efficient_netb2,
                # 'efficientnetb3': efficient_netb3,
                # 'efficientnetb4': efficient_netb4,
                'efficientnetb5': efficient_netb5,
                }

    base_model = models_dict[BASE_MODEL_NAME]

    model = build_model_bottom(inp, base_model, DROPOUT_RATE, False)

    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)

    lr_sched = polynomial_decay_schedule(initial_lr=1e-3, power=5.0, epochs=EPOCHS)

    decay = 1e-3/EPOCHS
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss="BinaryCrossentropy", optimizer=opt, metrics=['accuracy', precision_m, recall_m])


    history = model.fit(train_gen,
                        epochs= EPOCHS,
                        verbose=1,
                        validation_data=valid_gen,
                        callbacks=[lr_sched])

    plot_precision_recall_accuracy(history, total_training_steps)

    model.save(model_name)
