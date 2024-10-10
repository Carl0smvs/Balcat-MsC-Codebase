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
import random
from datetime import datetime

## ENV VARS

env = Env()

CI = env.bool('CI')
TEST = env.bool('TEST')
CALIBER = env.str('CALIBER')
MODEL_EXTENSION = env.str('MODEL_EXTENSION')

SAMPLING_MODE = env.str('SAMPLING_MODE')
SMALL_DATASET = env.bool('SMALL_DATASET')
SMALL_TRAIN_SIZE = env.int('SMALL_TRAIN_SIZE')
SMALL_VALIDATION_SIZE = env.int('SMALL_VALIDATION_SIZE')

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
    IMG_SIZE = (28, 28)

# For now...
IMG_SHAPE = IMG_SIZE + (3,)

tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)

K = 3

os.makedirs(DEFAULT_PLOT_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_STATISTICS_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_MODELS_DIRECTORY, exist_ok=True)

print("Caliber: " + CALIBER)

if CI:
    (train_dataset, test_dataset, validation_dataset, class_names, model_names) = (
        dataset_utils.load_siamese_dataset_ci_cartridges(
            IMG_SIZE,
            CLASSES_CARTRIDGES_CSV_FILE,
            CLASSES_CASEIDS_JSON_FILE,
            DATASET_DIR,
            sampling_mode=SAMPLING_MODE
        )
    )
else:
    (train_dataset, test_dataset, validation_dataset, class_names) = (
        dataset_utils.load_siamese_dataset_dev(
            IMG_SIZE,
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

modelType = "SIAMESE"
print('************ Training Model {model} ************'.format(model=modelType))

start_train = time.time()

model = model_utils.build_siamese_model(IMG_SHAPE)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
    loss=model_utils.contrastive_loss(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.F1Score(average="weighted")
    ]
)

# mlflow.keras.autolog()

history = model.fit(
    train_dataset,
    epochs=TRAIN_EPOCHS,
    validation_data=validation_dataset,
#    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=2
)


binary_accuracy = history.history['binary_accuracy']
val_binary_accuracy = history.history['val_binary_accuracy']

f1_score = history.history['f1_score']
val_f1_score = history.history['val_f1_score']

loss = history.history['loss']
val_loss = history.history['val_loss']

end_train = time.time()

f = open(DEFAULT_STATISTICS_DIRECTORY + "training.txt", "w")
f.write("Training time: " + str(end_train - start_train) + '\n')
f.write("Training epochs: " + str(TRAIN_EPOCHS) + '\n')
f.close()

tf_utils.plot_metric_for_train_validation(binary_accuracy, val_binary_accuracy,
                                          DEFAULT_PLOT_DIRECTORY + modelType + "-bin-acc-training-validation.png",
                                          'Binary Accuracy during Training and Validation')
tf_utils.plot_metric_for_train_validation(f1_score, val_f1_score,
                                          DEFAULT_PLOT_DIRECTORY + modelType + "-f1-score-training-validation.png",
                                          'F1-Score during Training and Validation')
tf_utils.plot_metric_for_train_validation(loss, val_loss,
                                          DEFAULT_PLOT_DIRECTORY + modelType + "-loss-training-validation.png",
                                          'Loss during Training and Validation',
                                          plot_limits=(0, 5))

if CI:
    run_name = modelType + ' model for ' + CALIBER + ' caliber - ' + str(datetime.now())
    model_name = modelType + "-BALCAT-" + CALIBER

else:
    run_name = modelType + ' model with MNIST dataset - ' + str(datetime.now())
    model_name = modelType + "-MNIST"

mlflow.set_tag('mlflow.runName', run_name)
mlflow.keras.log_model(model, artifact_path="keras-model", custom_objects={'loss': model_utils.contrastive_loss()}, registered_model_name=model_name)

if TEST:
    mlflow.set_tag('mlflow.runName', "TEST/DEBUG RUN " + modelType + " - " + CALIBER)

mlflow.log_artifact("/media")
mlflow.end_run()
