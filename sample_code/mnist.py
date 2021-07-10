#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import argparse

import tensorflow as tf
from tensorflow.keras import Model, layers
from metrics import Metrics

### Hyper-parameters
print("Tensorflow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

batch_size = 1000
lr = 1e-2
input_shape = (28, 28, 1)
tiled_shape = (56, 56, 3)
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SAVER_DIR = "./saved_model"
EVAL_DIR = "./eval_result"

# Not used. tf.example is deprecated.
binary_location = os.path.join(os.path.dirname(__file__), "dataset", "binary")

(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()
image_train, image_test = image_train / 255.0, image_test / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((image_train, label_train)).shuffle(5000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((image_test, label_test)).batch(batch_size)


def cnn_block(input, id):
    x = layers.Conv2D(8, 6, activation='relu', name='conv_1_{}'.format(id), dtype=tf.float64)(input)
    x = layers.Conv2D(16, 6, activation='relu', name='conv_2_{}'.format(id))(x)
    x = layers.Conv2D(24, 6, activation='relu', name='conv_3_{}'.format(id))(x)
    x = layers.Conv2D(32, 6, activation='relu', name='conv_4_{}'.format(id))(x)
    return x

 
# Optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels, model):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    return predictions


def tile_images(x, is_res50v2):
    x = tf.expand_dims(x, axis=3)
    if is_res50v2:
        multiples = tf.constant([1, 2, 2, 3], tf.int32)
        return tf.tile(x, multiples)
    return x


def main(epoch, is_res50v2, model, checkpoint_epoch=10):

    metrics = Metrics(label_list, EVAL_DIR)

    for ep in range(epoch):
        inputs = []
        answers = []
        preds = []

        for x, y in train_ds:
            train_step(tile_images(x, is_res50v2), y, model)

        for x, y in test_ds:
            pred = test_step(tile_images(x, is_res50v2), y, model)

            if ep % checkpoint_epoch == 0:
                inputs.append(x)
                answers.append(y)
                preds.append(pred)

        template = 'epoch: {}, loss: {}, accuracy: {}, test loss: {}, test accuracy: {}'
        print(template.format(ep+1,
                             train_loss.result(),
                             train_accuracy.result()*100,
                             test_loss.result(),
                             test_accuracy.result()*100))

        if ep % checkpoint_epoch == 0:
            inputs = tf.concat(inputs, 0)
            answers = tf.concat(answers, 0)
            preds = tf.concat(preds, 0)

            metrics.json_writer(inputs, answers, preds, 'evaluation_results.json')
            tf.keras.models.save_model(model, SAVER_DIR)
            print('Save model success.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='Total epoch')
    parser.add_argument('--res50v2', type=bool, default=False, help='Use predefined keras resnet model.')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluation mode.')
    args = parser.parse_args()

    # Training task can be skipped with loading trained model.
    if args.eval:
        pass

    if args.res50v2:
        # Heavy model (Predefined keras model)
        model_keras = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=tiled_shape)
        xh = layers.Flatten()(model_keras.layers[-1].output)
        output = layers.Dense(10, activation='softmax', name='dense_final')(xh)

        model = Model(inputs=model_keras.input, outputs=output)
    else:
        # Light model (Simple cnn model)
        inputs = tf.keras.Input(shape=input_shape, name='image', dtype=tf.float32)
        x = cnn_block(inputs, '1')
        x = layers.Flatten()(x)
        # x = layers.Dense(30, activation='relu', name='dense_1')(x)
        outputs = layers.Dense(10, activation='softmax', name='dense_2')(x)

        model = Model(inputs=inputs, outputs=outputs, name='mnist')

    model.summary()
    tf.keras.backend.set_floatx('float64')
    main(args.epoch, args.res50v2, model)

