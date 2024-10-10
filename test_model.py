import numpy as np
import tensorflow as tf
import utils.utils as tf_utils
import utils.dataset_load as dataset_utils
import utils.models as model_utils
import mlflow
import os
import urllib3
from environs import Env
from mlflow import MlflowClient
from datetime import datetime


## ENV VARS

env = Env()

CI = env.bool('CI')
TEST = env.bool('TEST')
GENERIC_MODEL = env.bool('GENERIC_MODEL')
WAVELET_FAMILY = env.str('WAVELET_FAMILY')
CALIBER = env.str('CALIBER')
MODEL_TYPE = env.str('MODEL_TYPE')
MODEL_EXTENSION = env.str('MODEL_EXTENSION')

SAMPLING_MODE = env.str('SAMPLING_MODE')
SMALL_DATASET = env.bool('SMALL_DATASET')
SMALL_TRAIN_SIZE = env.int('SMALL_TRAIN_SIZE')
SMALL_VALIDATION_SIZE = env.int('SMALL_VALIDATION_SIZE')

PLOT_IMAGES_ENABLED = env.bool('PLOT_IMAGES_ENABLED')
TRAIN_MODELS = env.bool('TRAIN_MODELS')
FINETUNE_MODELS = env.bool('FINETUNE_MODELS')
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
    IMG_SIZE = (960, 960)
else:
    IMG_SIZE = (224, 224)

# For now...
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

tf.random.set_seed(1234)
np.random.seed(1234)

os.makedirs(DEFAULT_IMAGES_DIRECTORY, exist_ok=True)
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

if CI:
    if WAVELET_FAMILY != "None":
        run_name = MODEL_TYPE + ' model for ' + CALIBER + ' caliber - Evaluation - for family ' + WAVELET_FAMILY + ' - ' + str(datetime.now())
        model_name = MODEL_TYPE + "-BALCAT-" + CALIBER + '-' + WAVELET_FAMILY
    else:
        run_name = MODEL_TYPE + ' model for ' + CALIBER + ' caliber - Evaluation - ' + str(datetime.now())
        model_name = MODEL_TYPE + "-BALCAT-" + CALIBER
else:
    model_name = MODEL_TYPE + "-MNIST"

modelType = tf_utils.Model[MODEL_TYPE]

print(model_name)

model = client.get_latest_versions(model_name)[0]
model = mlflow.keras.load_model(model_uri=f"models:/{model.name}/{model.version}")

## Run tests
print('************ Testing Model {model} ************'.format(model=modelType))

actual = None
predicted = None

if EVAL_ENABLED:
    ## Test model
    f = open(DEFAULT_STATISTICS_DIRECTORY + modelType.name + "-evaluation.txt", "w")

    loss, cat_acc, top_3_acc, f1_score = model.evaluate(test_dataset, verbose=0)

    f.write(f'Overall Accuracy: {cat_acc:.3f}\n')
    f.write(f'Top 3 Accuracy: {top_3_acc:.3f}\n')
    f.write(f'F1 Score: {f1_score:.3f}\n')
    f.write(f'Overall Loss: {loss:.3f}\n\n')

    # Also calculate for train set to check if it's training well
    actual, predicted, actual_preds, predicted_preds = tf_utils.get_actual_predicted_labels(model, train_dataset, custom=CI)

    print("PLOT ROC CURVES")
    tf_utils.plot_all_roc_curves(actual, predicted_preds,
                                 DEFAULT_PLOT_DIRECTORY + modelType.name + "-train-all-roc-curves.png",
                                 class_names=class_names)

    print("PLOT CONF MATRIX")
    tf_utils.plot_confusion_matrix(actual, predicted, class_names, 'testing',
                                   DEFAULT_PLOT_DIRECTORY + modelType.name + "-train-confusion-matrix.png")


    actual, predicted, actual_preds, predicted_preds = tf_utils.get_actual_predicted_labels(model, test_dataset, custom=CI)

    print("PLOT ROC CURVES")
    tf_utils.plot_all_roc_curves(actual, predicted_preds, DEFAULT_PLOT_DIRECTORY + modelType.name + "-all-roc-curves.png", class_names=class_names)

    print("PLOT CONF MATRIX")
    tf_utils.plot_confusion_matrix(actual, predicted, class_names, 'testing',
                                   DEFAULT_PLOT_DIRECTORY + modelType.name + "-confusion-matrix.png")

    (f_prec, f_rec, f1, macro_prec, macro_rec, acc_micro_averaged) = tf_utils.calculate_classification_metrics(actual, predicted, class_names)

    print("EVAL FILE")
    overall_acc = 0
    for label in class_names:
        f.write(label + " metrics:\n")
        f.write("       Precision: " + str(f_prec[label]) + '\n')
        f.write("       Recall: " + str(f_rec[label]) + '\n')
        f.write("       F1-Score: " + str(f1[label]) + '\n')

    f.write(f'\nPrecision (Macro-Averaged): {macro_prec:.3f}\n')
    f.write(f'Recall (Macro-Averaged): {macro_prec:.3f}\n')
    f.write(f'Accuracy (Micro-Averaged): {acc_micro_averaged:.3f}\n')
    f.close()

if PLOT_IMAGES_ENABLED:
    cur_date = datetime.now().date()
    os.makedirs(DEFAULT_IMAGES_DIRECTORY + str(cur_date), exist_ok=True)
    if WAVELET_FAMILY != "None":
        print("PLOT IMAGES WAVELET TRANSFORMATION")
        for label in images_to_plot_test.keys():
            i = 0
            for img in images_to_plot_test[label]:
                i += 1
                tf_utils.plot_wavelet_processed_image(
                    img, label, IMG_SHAPE, WAVELET_FAMILY,
                    DEFAULT_PLOT_DIRECTORY + modelType.name + "-" + WAVELET_FAMILY + "-" + label + "-" + str(i) + ".png")

    print("PLOT GRADCAM")
    tf_utils.plot_images_and_gradcam_from_list_per_class(
        images_to_plot_test, model, modelType, class_names,
        DEFAULT_PLOT_DIRECTORY + modelType.name + "-test-dataset-predictions-gradcam.png")

    for label in images_to_plot_test.keys():
        i = 0
        for img in images_to_plot_test[label]:
            i += 1
            tf_utils.plot_image_and_gradcam(img, label, model, modelType, class_names,
                                            DEFAULT_PLOT_DIRECTORY + modelType.name + "-gradcam-" + label + "-" + str(i) + ".png")



    print("PLOT WRONG IMAGES")
    ## Retrieve wrong samples and plot those
    if actual is None:
        actual, predicted, _, _ = tf_utils.get_actual_predicted_labels(model, test_dataset, custom=CI)
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
        DEFAULT_PLOT_DIRECTORY + modelType.name + "-test-dataset-wrong-samples-gradcam.png")

if CI:
    mlflow.set_tag('mlflow.runName', run_name)
else:
    mlflow.set_tag('mlflow.runName', MODEL_TYPE + ' model with MNIST dataset - Evaluation - ' + str(datetime.now()))

if TEST:
    mlflow.set_tag('mlflow.runName', "TEST/DEBUG EVALUATE " + MODEL_TYPE + " - " + CALIBER)

mlflow.log_artifact("/media")
mlflow.end_run()
