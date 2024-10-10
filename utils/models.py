import tensorflow as tf
from . import utils


def build_model(model_type, img_shape, wavelet_family="None", generic=False):

    inputs = tf.keras.Input(shape=img_shape, name='input_layer')

    inputs = utils.get_process_input_from_model_type(model_type)(inputs)

    core_model = utils.build_core_model(
            utils.get_base_model_from_model_type(model_type),
            False,
            input_tensor=inputs
    )
    if model_type == utils.Model.MOBILE_NET_V3_SMALL:
        core_model = utils.build_core_model(
            utils.get_base_model_from_model_type(model_type),
            False,
            input_tensor=inputs,
            minimalistic=True
        )

    core_model.trainable = False

    if generic:
        global_averaged = tf.keras.layers.GlobalAveragePooling2D()(core_model.get_layer(utils.get_last_generic_layer_from_model_type(model_type)).output)
    else:
        global_averaged = tf.keras.layers.GlobalAveragePooling2D()(core_model.layers[-1].output)

    dropout = tf.keras.layers.Dropout(0.2)(global_averaged)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(dropout)

    return core_model, tf.keras.Model(inputs=core_model.input, outputs=outputs)

def build_siamese_model(img_shape):
    img_A_input = tf.keras.Input(shape=img_shape, name='img_A_input')
    img_B_input = tf.keras.Input(shape=img_shape, name='img_B_input')

    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (32, 32), activation="tanh"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, (16, 16), activation="tanh"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (12, 12), activation="tanh"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, (8, 8), activation="tanh"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(512, (5, 5), activation="tanh"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(1024, (2, 2), strides=(2, 2), activation="tanh"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='tanh')
    ])


    print('************ Inner CNN architecture ************')
    embedding_network = tf.keras.Model(img_A_input, tf.keras.layers.Dense(1, activation='sigmoid')(cnn(img_A_input)))
    embedding_network.summary()

    feature_vector_A = cnn(img_A_input)
    feature_vector_B = cnn(img_B_input)
    merge_layer = tf.keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
        [feature_vector_A, feature_vector_B]
    )
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(normal_layer)

    return tf.keras.Model(inputs=[img_A_input, img_B_input], outputs=output)


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss_calculation(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss_calculation

