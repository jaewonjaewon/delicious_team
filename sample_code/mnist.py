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

batch_size = 500
epoch = 100
lr = 1e-2
input_shape = (162)
label_list = [0, 1]
SAVER_DIR = "./saved_model"
EVAL_DIR = "./eval_result"

TRAIN_DATA_PATH = "./train_data/train_result_train.csv"
TEST_DATA_PATH = "./train_data/train_result_valid.csv"

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=batch_size, # Artificially small to make examples easier to show.
      label_name='Repurchase',
      na_value="",
      num_epochs=epoch,
      ignore_errors=True,
      **kwargs)
  return dataset

raw_train_data = get_dataset(TRAIN_DATA_PATH)
raw_test_data = get_dataset(TEST_DATA_PATH)

def pack(features, label):
  features_value = list(features.values())
  convert_float = [tf.cast(feat, tf.float32) for feat in features_value]
  return tf.stack(convert_float, axis=-1), label

packed_train_data = raw_train_data.map(pack)
packed_test_data = raw_test_data.map(pack)

for features, labels in packed_train_data.take(1):
  print(f'Batch size: {len(features.numpy())}')
  print(f'Column size: {len(features.numpy()[0])}')
  print()
  print(labels.numpy())

train_ds = packed_train_data.shuffle(500)
test_ds = packed_test_data

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


def main(epoch, is_res50v2, model, checkpoint_epoch=10):

    metrics = Metrics(label_list, EVAL_DIR)

    for ep in range(epoch):
        inputs = []
        answers = []
        preds = []

        for x, y in train_ds:
            train_step(x, y, model)

        for x, y in test_ds:
            pred = test_step(x, y, model)

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
    parser.add_argument('--epoch', type=int, default=epoch, help='Total epoch')
    parser.add_argument('--res50v2', type=bool, default=False, help='Use predefined keras resnet model.')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluation mode.')
    args = parser.parse_args()

    # Training task can be skipped with loading trained model.
    if args.eval:
        pass

    if args.res50v2:
        # Heavy model (Predefined keras model)
        model_keras = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
        xh = layers.Flatten()(model_keras.layers[-1].output)
        output = layers.Dense(10, activation='softmax', name='dense_final')(xh)

        model = Model(inputs=model_keras.input, outputs=output)
    else:
        # Light model (Simple cnn model)
        inputs = tf.keras.Input(shape=input_shape, name='image', dtype=tf.float32)
        # x = cnn_block(inputs, '1')
        x = layers.Flatten()(inputs)
        x = layers.Dense(30, activation='relu', name='dense_1')(x)
        outputs = layers.Dense(2, activation='softmax', name='dense_2')(x)

        model = Model(inputs=inputs, outputs=outputs, name='delicious')

    model.summary()
    tf.keras.backend.set_floatx('float64')
    main(args.epoch, args.res50v2, model)

