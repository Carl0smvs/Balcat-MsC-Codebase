import numpy as np
import tensorflow as tf
import utils.utils as tf_utils
import utils.dataset_load as dataset_utils
import utils.models as model_utils
import mlflow
import os
import urllib3
import random
from environs import Env
from mlflow import MlflowClient
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

PLOT_IMAGES_ENABLED = env.bool('PLOT_IMAGES_ENABLED')
TRAIN_MODELS = env.bool('TRAIN_MODELS')
EVAL_ENABLED = env.bool('EVAL_ENABLED')

DATASET_DIR = env.str('DEFAULT_DATASET_DIRECTORY')
CLASSES_CARTRIDGES_CSV_FILE = env.str('CLASSES_CARTRIDGES_CSV_FILE')
CLASSES_CASEIDS_JSON_FILE = env.str('CLASSES_CASEIDS_JSON_FILE')

DEFAULT_IMAGES_DIRECTORY = env.str('DEFAULT_IMAGES_DIRECTORY')
DEFAULT_PLOT_DIRECTORY = env.str('DEFAULT_PLOT_DIRECTORY')
DEFAULT_STATISTICS_DIRECTORY = env.str('DEFAULT_STATISTICS_DIRECTORY')
DEFAULT_MODELS_DIRECTORY = env.str('DEFAULT_MODELS_DIRECTORY')

BASE_LEARNING_RATE = env.float('BASE_LEARNING_RATE')

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

## Disable mlflow console warnings from insecure TLS
urllib3.disable_warnings()

## Dataset Acquisition

BATCH_SIZE = 1

if CI:
    IMG_SIZE = (480, 480)
else:
    IMG_SIZE = (28, 28)

# For now...
IMG_SHAPE = IMG_SIZE + (3,)

tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)

os.makedirs(DEFAULT_IMAGES_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_PLOT_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_STATISTICS_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_MODELS_DIRECTORY, exist_ok=True)

print("Caliber: " + CALIBER)

if CI:
    (train_dataset, test_dataset, validation_dataset, train_class_pairs, val_class_pairs, test_class_pairs, class_names, model_names) = (
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

## Images to plot

if PLOT_IMAGES_ENABLED:
    images_to_plot_test = dict()
    for label in class_names:
        images_to_plot_test[label] = []
    for image, label in test_dataset:
        label = class_names[tf.argmax(label)]
        if len(images_to_plot_test[label]) == 10:
            continue
        images_to_plot_test[label].append(image)
        cont = False
        for l in images_to_plot_test.keys():
            if len(images_to_plot_test[l]) != 10:
                cont = True
                break
        if not cont:
            break

    ts = test_dataset

train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

validation_dataset = validation_dataset.batch(batch_size=BATCH_SIZE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

## MLFLOW Model Loading

mlflow.set_tracking_uri('url')
if CI:
    mlflow.set_experiment("BALCAT - CI")
else:
    mlflow.set_experiment("MNIST - DEV")
mlflow.start_run()

client = MlflowClient()

modelType = "SIAMESE"

if CI:
    run_name = modelType + ' model for ' + CALIBER + ' caliber - Evaluation - ' + str(datetime.now())
    model_name = modelType + "-BALCAT-" + CALIBER
else:
    model_name = modelType + "-MNIST"


print(model_name)

model = client.get_latest_versions(model_name)[0]

custom_objects = {'contrastive_loss_calculation': model_utils.contrastive_loss()}

with tf.keras.saving.custom_object_scope(custom_objects):
    model = mlflow.keras.load_model(model_uri=f"models:/{model.name}/{model.version}")

## Run tests
print('************ Testing Model {model} ************'.format(model=modelType))

actual = None
predicted = None

if EVAL_ENABLED:
    ## Test model
    f = open(DEFAULT_STATISTICS_DIRECTORY + modelType + "-evaluation.txt", "w")

    loss, bin_acc, f1_score = model.evaluate(test_dataset, verbose=0)
    f.write(f'Overall Accuracy: {bin_acc:.3f}\n')
    f.write(f'F1 Score: {f1_score:.3f}\n')
    f.write(f'Overall Loss: {loss:.3f}\n\n')

    # Also calculate for train set to check if it's training well
    actual, predicted, actual_preds, predicted_preds = tf_utils.get_actual_predicted_labels(model, train_dataset, siamese=True)

    #print("PLOT ROC CURVES")
    # tf_utils.plot_all_roc_curves(actual, predicted_preds,
    #                              DEFAULT_PLOT_DIRECTORY + modelType + "-train-all-roc-curves.png",
    #                              class_names=class_names)

    print("PLOT CONF MATRIX")
    tf_utils.plot_confusion_matrix(actual, predicted, class_names, 'testing',
                                   DEFAULT_PLOT_DIRECTORY + modelType + "-train-confusion-matrix.png")


    actual, predicted, actual_preds, predicted_preds = tf_utils.get_actual_predicted_labels(model, test_dataset, siamese=True)

    # print("PLOT ROC CURVES")
    # tf_utils.plot_all_roc_curves(actual, predicted_preds, DEFAULT_PLOT_DIRECTORY + modelType + "-all-roc-curves.png", class_names=class_names)

    print("PLOT CONF MATRIX")
    tf_utils.plot_confusion_matrix(actual, predicted, class_names, 'testing',
                                   DEFAULT_PLOT_DIRECTORY + modelType + "-confusion-matrix.png")

    pair_metrics = tf_utils.calculate_siamese_classification_metrics(actual, predicted, test_class_pairs, model_names)

    print("EVAL FILE")
    for pair in pair_metrics.keys():
        f.write(str(pair) + " metrics:\n")
        f.write("       TP: " + str(pair_metrics[pair]['tp']) + '\n')
        f.write("       FP: " + str(pair_metrics[pair]['fp']) + '\n')
        f.write("       FN: " + str(pair_metrics[pair]['fn']) + '\n')
        f.write("       TN: " + str(pair_metrics[pair]['tn']) + '\n')
        f.write("       Accuracy: " + str(pair_metrics[pair]['accuracy']) + '\n')
        f.write("       Precision: " + str(pair_metrics[pair]['precision']) + '\n')
        f.write("       Recall: " + str(pair_metrics[pair]['recall']) + '\n')
        f.write("       F1-Score: " + str(pair_metrics[pair]['f1_score']) + '\n')

    f.close()

if PLOT_IMAGES_ENABLED:

    # FIXME: review this part

    cur_date = datetime.now().date()
    os.makedirs(DEFAULT_IMAGES_DIRECTORY + str(cur_date), exist_ok=True)

    print("PLOT GRADCAM")
    tf_utils.plot_images_and_gradcam_from_list_per_class(
        images_to_plot_test, model, modelType, class_names,
        DEFAULT_IMAGES_DIRECTORY + str(cur_date) + "/" + CALIBER + "-" + modelType + "-test-dataset-predictions-gradcam.png")

    print("PLOT WRONG IMAGES")
    ## Retrieve wrong samples and plot those
    if actual is None:
        actual, predicted, _, _ = tf_utils.get_actual_predicted_labels(model, test_dataset, siamese=True)
    indexes_to_plot = []
    images_to_plot_wrong = dict()
    for i, (y, y_pred) in enumerate(zip(actual, predicted)):
        if y != y_pred:
            indexes_to_plot.append(i)
            if len(indexes_to_plot) == 10:
                break

    i_max = len(indexes_to_plot)
    cur = 0
    for i, (image, label) in enumerate(ts):
        if i not in indexes_to_plot:
            continue
        else:
            cur += 1
            label = class_names[tf.argmax(label)]
            if label in images_to_plot_wrong.keys():
                images_to_plot_wrong[label].append(image)
            else:
                images_to_plot_wrong[label] = [image]
            if cur == i_max:
                break
    tf_utils.plot_images_and_gradcam_from_list(
        images_to_plot_wrong, model, modelType, class_names, len(indexes_to_plot),
        DEFAULT_IMAGES_DIRECTORY + str(cur_date) + "/" + CALIBER + "-" + modelType + "-test-dataset-wrong-samples-gradcam.png")

if CI:
    mlflow.set_tag('mlflow.runName', run_name)
else:
    mlflow.set_tag('mlflow.runName', modelType + ' model with MNIST dataset - Evaluation - ' + str(datetime.now()))

if TEST:
    mlflow.set_tag('mlflow.runName', "TEST/DEBUG EVALUATE " + modelType + " - " + CALIBER)

mlflow.log_artifact("/media")
mlflow.end_run()
