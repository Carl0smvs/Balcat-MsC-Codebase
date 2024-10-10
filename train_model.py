import tensorflow as tf
import utils.utils as tf_utils
import utils.dataset_load as dataset_utils
import utils.models as model_utils
import numpy as np
from environs import Env
import time
import mlflow
import os
import urllib3
from datetime import datetime

## ENV VARS

env = Env()

CI = env.bool('CI')
TEST = env.bool('TEST')
GENERIC_MODEL = env.bool('GENERIC_MODEL')
CALIBER = env.str('CALIBER')
MODEL_TYPE = env.str('MODEL_TYPE')
MODEL_EXTENSION = env.str('MODEL_EXTENSION')

SAMPLING_MODE = env.str('SAMPLING_MODE')
WAVELET_FAMILY = env.str('WAVELET_FAMILY')
SMALL_DATASET = env.bool('SMALL_DATASET')
SMALL_TRAIN_SIZE = env.int('SMALL_TRAIN_SIZE')
SMALL_VALIDATION_SIZE = env.int('SMALL_VALIDATION_SIZE')

FINETUNE_MODELS = env.bool('FINETUNE_MODELS')

DEFAULT_PLOT_DIRECTORY = env.str('DEFAULT_PLOT_DIRECTORY')
DEFAULT_STATISTICS_DIRECTORY = env.str('DEFAULT_STATISTICS_DIRECTORY')
DEFAULT_MODELS_DIRECTORY = env.str('DEFAULT_MODELS_DIRECTORY')
DATASET_DIR = env.str('DEFAULT_DATASET_DIRECTORY')

CLASSES_CARTRIDGES_CSV_FILE = env.str('CLASSES_CARTRIDGES_CSV_FILE')
CLASSES_CASEIDS_JSON_FILE = env.str('CLASSES_CASEIDS_JSON_FILE')

TRAIN_EPOCHS = env.int('TRAIN_EPOCHS')
FINETUNE_EPOCHS = env.int('FINETUNE_EPOCHS')
BASE_LEARNING_RATE = env.float('BASE_LEARNING_RATE')

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"


## Disable mlflow console warnings from insecure TLS
urllib3.disable_warnings()

## Dataset Acquisition

BATCH_SIZE = 8

if CI:
    IMG_SIZE = (480, 480)
else:
    IMG_SIZE = (224, 224)

# For now...
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

tf.random.set_seed(1234)
np.random.seed(1234)

K = 3

os.makedirs(DEFAULT_PLOT_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_STATISTICS_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_MODELS_DIRECTORY, exist_ok=True)

if CI:
    (train_dataset, test_dataset, validation_dataset, class_names) = (
        dataset_utils.load_dataset_ci_cartridges(
            IMG_SIZE,
            CLASSES_CARTRIDGES_CSV_FILE,
            CLASSES_CASEIDS_JSON_FILE,
            DATASET_DIR,
            sampling_mode=SAMPLING_MODE
        )
    )
else:
    (train_dataset, test_dataset, validation_dataset, class_names) = (
        dataset_utils.load_dataset_dev(IMG_SIZE,
                                       small_dataset=SMALL_DATASET,
                                       train_size=SMALL_TRAIN_SIZE,
                                       val_size=SMALL_VALIDATION_SIZE
                                       )
    )

train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

validation_dataset = validation_dataset.batch(batch_size=BATCH_SIZE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

## MLFLOW Tracking

mlflow.set_tracking_uri('url')
if CI:
    mlflow.set_experiment("BALCAT - CI")
else:
    mlflow.set_experiment("MNIST - DEV")
mlflow.start_run()

modelType = tf_utils.Model[MODEL_TYPE]
print('************ Training Model {model} ************'.format(model=modelType))

start_train = time.time()

if FINETUNE_MODELS:
    BASE_LEARNING_RATE /= 100

(core_model, finished_model) = model_utils.build_model(modelType, IMG_SHAPE, wavelet_family=WAVELET_FAMILY, generic=GENERIC_MODEL)
finished_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
                       loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=[
                                tf.keras.metrics.CategoricalAccuracy(),
                                tf.keras.metrics.TopKCategoricalAccuracy(k=K),
                                tf.keras.metrics.F1Score(average="weighted")
                                ])

# Unfreeze the pre-trained layers
if FINETUNE_MODELS:
    for layer in core_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


mlflow.keras.autolog()

history = finished_model.fit(train_dataset,
                             epochs=TRAIN_EPOCHS,
                             validation_data=validation_dataset,
                             callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
                             verbose=2
                             )

finished_model.summary()

categorical_accuracy = history.history['categorical_accuracy']
val_categorical_accuracy = history.history['val_categorical_accuracy']

top_k_categorical_accuracy = history.history['top_k_categorical_accuracy']
val_top_k_categorical_accuracy = history.history['val_top_k_categorical_accuracy']

f1_score = history.history['f1_score']
val_f1_score = history.history['val_f1_score']

loss = history.history['loss']
val_loss = history.history['val_loss']

end_train = time.time()

f = open(DEFAULT_STATISTICS_DIRECTORY + "training.txt", "w")
f.write("Training time: " + str(end_train - start_train) + '\n')
f.write("Training epochs: " + str(TRAIN_EPOCHS) + '\n')
f.close()

tf_utils.plot_metric_for_train_validation(categorical_accuracy, val_categorical_accuracy,
                                          DEFAULT_PLOT_DIRECTORY + modelType.name + "-cat-acc-training-validation.png",
                                          'Categorical Accuracy during Training and Validation')
tf_utils.plot_metric_for_train_validation(top_k_categorical_accuracy, val_top_k_categorical_accuracy,
                                          DEFAULT_PLOT_DIRECTORY + modelType.name + "-top-acc-training-validation.png",
                                          'Top ' + str(K) + ' Accuracy during Training and Validation')
tf_utils.plot_metric_for_train_validation(f1_score, val_f1_score,
                                          DEFAULT_PLOT_DIRECTORY + modelType.name + "-f1-score-training-validation.png",
                                          'F1-Score during Training and Validation')
tf_utils.plot_metric_for_train_validation(loss, val_loss,
                                          DEFAULT_PLOT_DIRECTORY + modelType.name + "-loss-training-validation.png",
                                          'Loss during Training and Validation',
                                          plot_limits=(0, 5))

if CI:
    if WAVELET_FAMILY != "None":
        run_name = MODEL_TYPE + ' model for ' + CALIBER + ' caliber for family ' + WAVELET_FAMILY + ' - ' + str(datetime.now())
        model_name = MODEL_TYPE + "-BALCAT-" + CALIBER + '-' + WAVELET_FAMILY
    else:
        run_name = MODEL_TYPE + ' model for ' + CALIBER + ' caliber - ' + str(datetime.now())
        model_name = MODEL_TYPE + "-BALCAT-" + CALIBER

else:
    run_name = MODEL_TYPE + ' model with MNIST dataset - ' + str(datetime.now())
    model_name = MODEL_TYPE + "-MNIST"

mlflow.set_tag('mlflow.runName', run_name)
mlflow.keras.log_model(finished_model, artifact_path="keras-model", registered_model_name=model_name)

if TEST:
    mlflow.set_tag('mlflow.runName', "TEST/DEBUG RUN " + MODEL_TYPE + " - " + CALIBER)

mlflow.log_artifact("/media")
mlflow.end_run()
