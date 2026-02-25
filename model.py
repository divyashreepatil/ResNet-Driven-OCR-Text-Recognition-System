import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x, filters, downsample=False):
    """Residual block for ResNet architecture"""
    shortcut = x
    strides = 1

    # Check if downsampling is needed OR if the number of filters is changing
    if downsample or x.shape[-1] != filters:
        strides = 2 if downsample else 1
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, (3, 3), strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x


def build_resnet_model(input_shape=(32, 32, 1), num_classes=10):
    """Build ResNet-based CNN model for OCR"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Add residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32, downsample=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


def load_and_prepare_data():
    """Load and prepare MNIST data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0,1] and expand channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    # Resize to (32,32,1) for ResNet compatibility
    x_train = tf.image.resize(x_train, [32, 32])
    x_test = tf.image.resize(x_test, [32, 32])

    return (x_train, y_train), (x_test, y_test)


def train_model(model=None, epochs=5, batch_size=128):
    """Train the model on MNIST data"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    if model is None:
        model = build_resnet_model()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, history


def save_model(model, path='ocr_model.h5'):
    """Save the trained model"""
    model.save(path)
    print(f"Model saved to {path}")


def load_trained_model(path='ocr_model.h5'):
    """Load a pre-trained model"""
    return tf.keras.models.load_model(path)
