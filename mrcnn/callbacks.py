"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras


__author__ = 'roeiherz'


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


class Evaluate(keras.callbacks.Callback):
    """
    Evaluation callback for arbitrary datasets.
    """

    def __init__(self, dataset, iou_threshold=0.5, save_path=None, tensorboard=None, verbose=1, config=None):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            dataset       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.dataset = dataset
        self.iou_threshold = iou_threshold
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.verbose = verbose
        self.config = config

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        from samples.bdd100k.BDD100K import evaluate
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.dataset,
            self.model,
            iou_threshold=self.iou_threshold,
            save_path=self.save_path,
            config=self.config
        )

        self.mean_ap = sum(average_precisions.values()) / len(average_precisions)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            for label, average_precision in average_precisions.items():
                print(self.dataset.class_names[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(self.mean_ap))
