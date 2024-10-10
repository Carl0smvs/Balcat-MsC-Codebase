import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.keras.applications as apps
import scikitplot.metrics as plot_metrics
from scipy import interp

import tensorflow_wavelets.Layers.DWT as DWT
import keras_cv

from enum import Enum

# from . import models


class Model(Enum):
    EFFICIENT_NET_B0 = 3
    VGG_16 = 2
    XCEPTION = 1
    INCEPTION_V3 = 4
    DENSE_NET_121 = 5
    MOBILE_NET_V2 = 6
    MOBILE_NET_V3_SMALL = 7
    SIAMESE = 8


def build_core_model(model_builder, include_top, weights='imagenet', input_tensor=None, minimalistic=None):
    model = model_builder(include_top=include_top,
                          weights=weights,
                          input_tensor=input_tensor)

    if minimalistic is not None:
        model = model_builder(include_top=include_top,
                              weights=weights,
                              input_tensor=input_tensor,
                              minimalistic=minimalistic)

    return model

def build_core_cnn_model():
    return None


def build_final_model(core_model, average_layer, dropout_layer, prediction_layer):
    x = average_layer(core_model.layers[-1].output)
    x = dropout_layer(x)
    outputs = prediction_layer(x)

    return tf.keras.Model(inputs=core_model.input, outputs=outputs)


def get_model_from_model_type(model_type: Model, image_size, num_classes):
    # TODO: SNN

    inputs = tf.keras.Input(shape=image_size + (3,), name='input_layer')

    if model_type != Model.SIAMESE:
        inputs = get_process_input_from_model_type(model_type)(inputs)
    else:
        # FIXME: TBD
        a = 0


    core_model = build_core_model(
        get_base_model_from_model_type(model_type),
        False,
        input_tensor=inputs)

    if model_type == Model.MOBILE_NET_V3_SMALL:
        core_model = build_core_model(
            get_base_model_from_model_type(model_type),
            False,
            input_tensor=inputs,
            minimalistic=True
        )

    core_model.trainable = False

    predictions_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    ## Assemble model
    finished_model = build_final_model(
        core_model,
        global_average_layer,
        tf.keras.layers.Dropout(0.5),
        predictions_layer)

    return core_model, finished_model

def get_base_model_from_model_type(model_type: Model):
    if model_type == Model.EFFICIENT_NET_B0:
        return apps.efficientnet.EfficientNetB0
    if model_type == Model.VGG_16:
        return apps.vgg16.VGG16
    if model_type == Model.XCEPTION:
        return apps.xception.Xception
    if model_type == Model.INCEPTION_V3:
        return apps.inception_v3.InceptionV3
    if model_type == Model.DENSE_NET_121:
        return apps.densenet.DenseNet121
    if model_type == Model.MOBILE_NET_V2:
        return apps.mobilenet_v2.MobileNetV2
    if model_type == Model.MOBILE_NET_V3_SMALL:
        return apps.MobileNetV3Small
    if model_type == Model.SIAMESE:
        return None


def get_process_input_from_model_type(model_type: Model):
    if model_type == Model.EFFICIENT_NET_B0:
        return apps.efficientnet.preprocess_input
    if model_type == Model.VGG_16:
        return apps.vgg16.preprocess_input
    if model_type == Model.XCEPTION:
        return apps.xception.preprocess_input
    if model_type == Model.INCEPTION_V3:
        return apps.inception_v3.preprocess_input
    if model_type == Model.DENSE_NET_121:
        return apps.densenet.preprocess_input
    if model_type == Model.MOBILE_NET_V2:
        return apps.mobilenet_v2.preprocess_input
    if model_type == Model.MOBILE_NET_V3_SMALL:
        return apps.mobilenet_v3.preprocess_input
    if model_type == Model.SIAMESE:
        return None


def get_decode_predictions_from_model_type(model_type: Model):
    if model_type == Model.EFFICIENT_NET_B0:
        return apps.efficientnet.decode_predictions
    if model_type == Model.VGG_16:
        return apps.vgg16.decode_predictions
    if model_type == Model.XCEPTION:
        return apps.xception.decode_predictions
    if model_type == Model.INCEPTION_V3:
        return apps.inception_v3.decode_predictions
    if model_type == Model.DENSE_NET_121:
        return apps.densenet.decode_predictions
    if model_type == Model.MOBILE_NET_V2:
        return apps.mobilenet_v2.decode_predictions
    if model_type == Model.MOBILE_NET_V3_SMALL:
        return apps.mobilenet_v3.decode_predictions
    if model_type == Model.SIAMESE:
        return None


def get_last_generic_layer_from_model_type(model_type: Model):
    # TODO: the rest
    if model_type == Model.EFFICIENT_NET_B0:
        return 'top_activation'
    if model_type == Model.VGG_16:
        return 'block5_conv3'
    if model_type == Model.XCEPTION:
        return 'block14_sepconv2_act'
    if model_type == Model.INCEPTION_V3:
        return 'mixed10'
    if model_type == Model.DENSE_NET_121:
        return 'conv5_block16_concat'
    if model_type == Model.MOBILE_NET_V2:
        #return 'block_8_expand_relu' # convolution, normalization and relu of outputs of block 7/16
        return 'block_3_expand_relu'
    if model_type == Model.MOBILE_NET_V3_SMALL:
        return 'Conv_1'
    if model_type == Model.SIAMESE:
        # FIXME: Specify the last conv layer here
        return 'TBD'


## Adjust these better for each net. EfficientNet is wrong
def get_last_conv_layer_from_model_type(model_type: Model):
    if model_type == Model.EFFICIENT_NET_B0:
        return 'top_activation'
    if model_type == Model.VGG_16:
        return 'block5_conv3'
    if model_type == Model.XCEPTION:
        return 'block14_sepconv2_act'
    if model_type == Model.INCEPTION_V3:
        return 'mixed10'
    if model_type == Model.DENSE_NET_121:
        return 'conv5_block16_concat'
    if model_type == Model.MOBILE_NET_V2:
        return 'Conv_1'
    if model_type == Model.MOBILE_NET_V3_SMALL:
        return 'Conv_1'
    if model_type == Model.SIAMESE:
        # FIXME: Specify the last conv layer here
        return 'TBD'


def plot_dataset_distribution_piechart(dataset, labels, filename):
    def percentage_format(percentage):
        return '{:.2f}%\n({:.0f})'.format(percentage, total * percentage / 100)

    class_count = np.zeros(len(labels), dtype=np.int32)
    for _, label in dataset:
        class_count[tf.argmax(label)] += 1

    total = np.sum(class_count)
    explode = np.ones(len(labels)) * 0.05
    fig1, ax1 = plt.subplots()

    plt.pie(class_count, labels=labels, autopct=percentage_format, startangle=90, pctdistance=0.85,
            explode=explode)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()

    plt.savefig(filename)
    plt.cla()
    plt.clf()

def plot_pychart(labels, counts, title, filename):
    def percentage_format(percentage):
        return '{:.2f}%\n({:.0f})'.format(percentage, total * percentage / 100)

    total = np.sum(counts)
    explode = np.ones(len(labels)) * 0.05
    fig1, ax1 = plt.subplots()

    plt.pie(counts, labels=labels, autopct=percentage_format, startangle=90, pctdistance=0.85,
            explode=explode)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.title(title)

    plt.savefig(filename)
    plt.cla()
    plt.clf()


def plot_accuracies_losses_training_fine_tuning(acc, val_acc, loss, val_loss, initial_epochs, filepath):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.ylim((0, 1))
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(filepath)
    plt.cla()
    plt.clf()

def plot_metric_for_train_validation(train, validation, filepath, title, plot_limits=(0, 1), epochs_split=None):
    #plt.figure(figsize=(8, 8))
    #plt.subplot(2, 1, 1)
    plt.plot(train, label='Training Time')
    plt.plot(validation, label='Validation Time')
    if epochs_split is not None:
        plt.plot([epochs_split, epochs_split],
                plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.ylim(plot_limits)
    plt.title(title)
    plt.savefig(filepath)
    plt.cla()
    plt.clf()


def get_model_layer_index(model, layer_name):
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return idx


def get_img_array(image):
    array = keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(orig_img, heatmap, image_path="", alpha=0.4, save=True):

    img_array = keras.utils.img_to_array(orig_img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    if save:
        superimposed_img.save(image_path)

    # Display Grad CAM
    # Image.open(image_path).show()

    return superimposed_img


def get_actual_predicted_labels(model, dataset, custom=False, siamese=False):
    actual = []
    if custom:
        for _, label in dataset:
            actual.append(label)
    else:
        actual = [labels for _, labels in dataset.unbatch()]

    predicted = model.predict(dataset, verbose=0)

    if siamese:
        actual_preds = actual
        #actual = tf.math.round(actual)
        predicted_preds = predicted
        predicted = tf.math.round(predicted)
        return actual, predicted, actual_preds, predicted_preds
    else:
        actual = tf.stack(actual, axis=0)
        actual_preds = actual
        print(actual)
        print(actual_preds)
        actual = tf.argmax(actual, axis=2)
        predicted_preds = predicted
        predicted = tf.argmax(predicted, axis=1)

        return actual, predicted, actual_preds, predicted_preds


def get_predicted_labels_from_list(model, l):
    predicted = [model.predict(x[None, ...], verbose=0) for x in l]
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return predicted


def plot_confusion_matrix(actual, predicted, labels, ds_type, filepath):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig(filepath)
    plt.cla()
    plt.clf()


def plot_images_and_gradcam_from_list_per_class(images_to_plot, model, modelType, class_names, filename):
    n_rows = len(images_to_plot)
    n_cols = int(min([len(images_to_plot[x]) for x in class_names]) / 2)

    ## Get the predicted labels of the sample test dataset
    labels_to_plot_test_gradcam = dict()
    for label in class_names:
        predicted = get_predicted_labels_from_list(model, images_to_plot[label])
        labels_to_plot_test_gradcam[label] = [class_names[int(x)] for x in predicted]

        ## Generate superimposed gradcam images
    preprocess_input = get_process_input_from_model_type(modelType)
    target_layer = get_last_conv_layer_from_model_type(modelType)

    model.layers[-1].activation = None

    images_to_plot_gradcam = dict()
    for label in class_names:
        images_to_plot_gradcam[label] = []

    for i in range(n_rows * n_cols):
        label = class_names[i // n_cols]
        img_array = preprocess_input(
            get_img_array(images_to_plot[label][i % n_cols].numpy().astype("uint8")))

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, target_layer)

        ## Superimposed image
        images_to_plot_gradcam[label].append(
            save_and_display_gradcam(images_to_plot[label][i % n_cols], heatmap, alpha=0.7, save=False))

    fig = plt.figure(constrained_layout=True, figsize=(4 * n_cols, 2 * n_rows))
    subfigs = fig.subfigures(n_rows, n_cols)

    for i, subfig in enumerate(subfigs.flat):

        label = class_names[i // n_cols]
        pred_label = labels_to_plot_test_gradcam[label][i % n_cols]
        images = [images_to_plot[label][i % n_cols],
                  images_to_plot_gradcam[label][i % n_cols]]  # second should be gradcam

        if label == pred_label:
            font_color = 'green'
        else:
            font_color = 'red'

        subfig.supxlabel("Predicted: " + pred_label, rotation=0, size='large',
                         ha='center', va='center', color=font_color)
        if i % n_cols == 0:
            subfig.supylabel("Class " + label, rotation=0, size='large',
                             ha='left', va='center')
        else:
            subfig.supylabel("", rotation=0, size='large',
                             ha='left', va='center')
        axs = subfig.subplots(1, 2)
        for j, ax in enumerate(axs.flat):
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            if j == 0:
                ax.imshow(images[j].numpy().astype("uint8"))
            else:
                ax.imshow(images[j])

    plt.savefig(filename)
    plt.cla()
    plt.clf()


def plot_images_and_gradcam_from_list(images_to_plot, model, modelType, class_names, number_of_images, filename):
    n_rows = 1
    n_cols = number_of_images

    ## Get the predicted labels of the sample test dataset
    orig_labels = []
    orig_images = []

    for label in images_to_plot.keys():
        for image in images_to_plot[label]:
            orig_labels.append(label)
            orig_images.append(image)


    predicted = get_predicted_labels_from_list(model, orig_images)
    pred_labels = [class_names[int(x)] for x in predicted]


    ## Generate superimposed gradcam images
    preprocess_input = get_process_input_from_model_type(modelType)
    target_layer = get_last_conv_layer_from_model_type(modelType)

    model.layers[-1].activation = None

    gradcam_images = []
    for image in orig_images:
        img_array = preprocess_input(
            get_img_array(image.numpy().astype("uint8")))

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, target_layer)

        ## Superimposed image
        gradcam_images.append(save_and_display_gradcam(image, heatmap, alpha=0.7, save=False))

    fig = plt.figure(constrained_layout=True, figsize=(4 * n_cols, 2 * n_rows))
    subfigs = fig.subfigures(n_rows, n_cols)

    for i, subfig in enumerate(subfigs.flat):
        label = orig_labels[i]
        pred_label = pred_labels[i]
        images = [orig_images[i], gradcam_images[i]]

        subfig.supxlabel("Predicted: " + pred_label + "\nActual Label: " + label, rotation=0, size='large', ha='center', va='bottom')

        axs = subfig.subplots(1, 2)
        for j, ax in enumerate(axs.flat):
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            if j == 0:
                ax.imshow(images[j].numpy().astype("uint8"))
            else:
                ax.imshow(images[j])

    plt.savefig(filename)
    plt.cla()
    plt.clf()

def plot_image_and_gradcam(image, label, model, modelType, class_names, filename):

    predicted_label = class_names[int(get_predicted_labels_from_list(model, [image])[0])]

    ## Generate superimposed gradcam images
    preprocess_input = get_process_input_from_model_type(modelType)
    target_layer = get_last_conv_layer_from_model_type(modelType)

    model.layers[-1].activation = None

    img_array = preprocess_input(
        get_img_array(image.numpy().astype("uint8")))
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, target_layer)
    superimposed_img = save_and_display_gradcam(image, heatmap, alpha=0.7, image_path=filename)

    fig = plt.figure(constrained_layout=True, figsize=(4, 2))

    images = [image, superimposed_img]

    fig.supxlabel("Predicted: " + predicted_label + "\nActual Label: " + label, rotation=0, size='large', ha='center',
                     va='bottom')

    axs = fig.subplots(1, 2)
    for j, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        if j == 0:
            ax.imshow(images[j].numpy().astype("uint8"))
        else:
            ax.imshow(images[j])

    plt.savefig(filename)
    plt.cla()
    plt.clf()


def plot_all_roc_curves(y_true, y_pred_probs, filename, class_names=None):
    plot_roc(y_true, y_pred_probs, class_names=class_names)
    plt.savefig(filename)
    plt.cla()
    plt.clf()


def plot_roc(y_true, y_probas,
             title='ROC Curves',
             plot_micro=True,
             plot_macro=True,
             classes_to_plot=None,
             class_names=None,
             ax=None,
             figsize=None,
             cmap='nipy_spectral',
             title_fontsize="large",
             text_fontsize="medium"
             ):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        plot_macro (boolean, optional): Plot the macro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``

        class_names (list-like, optional): Class names for which the ROC
            curve should be plotted. e.g. [0, 'cold']. Must be the same shape as clases_too_plot.
             Defaults to ``None``

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = plot_metrics.roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            roc_auc = plot_metrics.auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                    label='{0} (area = {1:0.2f})'
                          ''.format(class_names[i], roc_auc))

    if plot_micro:
        binarized_y_true = plot_metrics.label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = plot_metrics.roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = plot_metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='deeppink', linestyle=':', linewidth=4)

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = plot_metrics.auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr,
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


def calculate_classification_metrics(y_actual, y_pred, labels):
    """
    Calculate the precision and recall of a classification model using the ground truth and
    predicted values.

    Args:
      y_actual: Ground truth labels.
      y_pred: Predicted labels.
      labels: List of classification labels.

    Return:
      Precision and recall measures.
    """
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    tp = np.diag(cm) # Diagonal represents true positives
    matrix_sum = np.sum(cm)

    precision = dict()
    recall = dict()
    f1_score = dict()

    prec_all = 0
    rec_all = 0

    tp_all = 0
    fp_all = 0
    for i in range(len(labels)):
        col = cm[:, i]
        fp = np.sum(col) - tp[i] # Sum of column minus true positive is false negative

        tp_all += tp[i]
        fp_all += fp

        row = cm[i, :]
        fn = np.sum(row) - tp[i] # Sum of row minus true positive, is false negative

        prec = tp[i] / (tp[i] + fp)
        prec_all += prec
        precision[labels[i]] = prec

        rec = tp[i] / (tp[i] + fn)
        rec_all += rec
        recall[labels[i]] = rec
        f1_score[labels[i]] = 2 * tp[i] / (2 * tp[i] + fp + fn)

    return precision, recall, f1_score, prec_all / len(labels), rec_all / len(labels), tp_all / (tp_all + fp_all)


def calculate_siamese_classification_metrics(y_actual, y_pred, labels, class_names):

    # Make sure the pairs are in the same order
    labels = [x if class_names.index(x[0]) < class_names.index(x[1]) else (x[1], x[0]) for x in labels]

    possible_pairs = []
    for name1 in class_names:
        for name2 in class_names:
            possible_pairs.append((name1, name2))
    correspondence_matrix = np.zeros((len(possible_pairs), 4))
    for actual, pred, true_pair in zip(y_actual, y_pred, labels):
        index = 2 * actual.numpy().astype("uint8") + pred.numpy().astype("uint8")
        correspondence_matrix[possible_pairs.index(true_pair)][index] += 1

    pair_metrics = dict()
    # We want to compute metrics per pair
    for i, pair in enumerate(possible_pairs):
        pair_metrics[str(pair)] = dict()
        tp, fn, fp, tn = correspondence_matrix[i]
        pair_metrics[str(pair)]['tp'] = tp
        pair_metrics[str(pair)]['fn'] = fn
        pair_metrics[str(pair)]['fp'] = fp
        pair_metrics[str(pair)]['tn'] = tn
        pair_metrics[str(pair)]['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        pair_metrics[str(pair)]['precision'] = (tp) / (tp + fp)
        pair_metrics[str(pair)]['recall'] = (tp) / (tp + fn)
        pair_metrics[str(pair)]['f1_score'] = (2 * pair_metrics[str(pair)]['precision'] * pair_metrics[str(pair)]['recall']) / (pair_metrics[str(pair)]['precision'] + pair_metrics[str(pair)]['recall'])

    return pair_metrics

def plot_wavelet_processed_image(image, label, image_shape, wavelet_family, filename):
    inputs = tf.keras.Input(shape=image_shape, name='input_layer')
    grayscale_inputs = keras_cv.layers.Grayscale()
    wavelet_decompositions = DWT.DWT(wavelet_family, 0)

    data_processing = tf.keras.Sequential([inputs, grayscale_inputs, wavelet_decompositions])
    result_image = tf.cast(tf.expand_dims(image, 0), tf.float32)

    result_image = data_processing(result_image)[0, :, :, :-1]

    fig = plt.figure(constrained_layout=True, figsize=(4, 2))

    images = [image, result_image]

    fig.supxlabel("Label: " + label, rotation=0, size='large', ha='center', va='bottom')

    axs = fig.subplots(1, 2)
    for j, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        if j == 0:
            ax.imshow(images[j].numpy().astype("uint8"))
        else:
            ax.imshow(images[j])

    plt.savefig(filename)
    plt.cla()
    plt.clf()



    tf.keras.utils.save_img(filename, result_image)