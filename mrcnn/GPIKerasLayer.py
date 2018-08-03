import tensorflow as tf
import keras.engine as KE


############################################################
#  Graph Permeation Invariant Layers
############################################################


class GPIKerasLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, num_rois, feature_size=1024, batch_size=1, gpi_type='FeatureAttention', **kwargs):
        super(GPIKerasLayer, self).__init__(**kwargs)
        self.num_rois = num_rois
        self.gpi_type = gpi_type
        self.batch_size = batch_size
        self.feature_size = feature_size

    def call(self, inputs):
        # Object features - [batch ,num_boxes , 1024]
        entity_features = inputs[0]
        # Relation features - [batch ,num_boxes^2 , 1024]
        relation_features = inputs[1]
        # Single ROIs - [batch, num_boxes , 4]
        single_rois = inputs[2]
        # Pairwise ROIs - [batch, num_boxes, num_boxes, 4]
        pairwise_rois = inputs[3]

        # Reshape [batch, num_boxes, num_boxes, 1024]
        relation_features_rs = tf.reshape(relation_features,
                                          (self.batch_size, self.num_rois, self.num_rois, self.feature_size))
        relation_features_pr = tf.transpose(relation_features_rs, perm=[0, 2, 1, 3])
        # Concat relations in the feature axis - [batch_num, num_boxes, num_boxes, feature_size * 2]
        relation_features = tf.concat([relation_features_rs, relation_features_pr], axis=3)
        # Concat spatial features - [batch, num_boxes, features_size + 4]
        entity_features = tf.concat([entity_features, single_rois], axis=2)
        # Reshape to [batch, num_boxes, features_size + 4]
        entity_features = tf.reshape(entity_features, (self.batch_size, self.num_rois, self.feature_size + 4))
        # Reshape to [batch, num_boxes, num_boxes, 4]
        shaped_relation_bb = tf.reshape(pairwise_rois, (self.batch_size, self.num_rois, self.num_rois, 4))
        # Concat to feature axis - [batch, num_boxes, num_boxes, feature_size * 2 + 4]
        relation_features = tf.concat([relation_features, shaped_relation_bb], axis=3)
        return [entity_features, relation_features]

    def compute_output_shape(self, input_shape):
        return [(self.batch_size, self.num_rois, self.feature_size + 4),
                (self.batch_size, self.num_rois, self.num_rois, self.feature_size * 2 + 4)]


class GPIKerasExpandLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, num_rois, feature_size=1024, batch_size=1, gpi_type='FeatureAttention', **kwargs):
        super(GPIKerasExpandLayer, self).__init__(**kwargs)
        self.num_rois = num_rois
        self.gpi_type = gpi_type
        self.batch_size = batch_size
        self.feature_size = feature_size

    def call(self, inputs):
        # Object features - [batch ,num_boxes , 1024]
        entity_features = inputs[0]
        # Relation features - [batch ,num_boxes^2 , 1024]
        relation_features = inputs[1]

        # Expand object confidence
        # [num_boxes, num_boxes, features_size + 4]
        extended_confidence_entity_shape = (self.num_rois, self.num_rois, entity_features.shape[2])
        # [num_boxes, num_boxes, features_size + 4]
        expand_object_features = tf.add(tf.zeros(extended_confidence_entity_shape), tf.squeeze(entity_features, 0),
                                        name="expand_object_features")

        # Expand subject confidence
        expand_subject_features = tf.transpose(expand_object_features, perm=[1, 0, 2], name="expand_subject_features")

        # Node Neighbours
        object_ngbrs = [expand_object_features, expand_subject_features, tf.squeeze(relation_features, 0)]
        object_ngbrs_tensor = tf.concat(object_ngbrs, axis=-1)

        return tf.expand_dims(object_ngbrs_tensor, 0)

    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.num_rois, self.num_rois, (self.feature_size + 4) * 2 + self.feature_size * 2 + 4)


class GPIKerasExpandGraphLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, num_rois, feature_size=1024, batch_size=1, gpi_type='FeatureAttention', **kwargs):
        super(GPIKerasExpandGraphLayer, self).__init__(**kwargs)
        self.num_rois = num_rois
        self.gpi_type = gpi_type
        self.batch_size = batch_size
        self.feature_size = feature_size

    def call(self, inputs):
        # Object features - [batch , 500]
        object_ngbrs_alpha_all = inputs[0]

        # Graph encoding - [batch, num_boxes, 500]
        expand_graph_sh = (self.batch_size, self.num_rois, self.feature_size)
        expand_graph = tf.add(tf.zeros(expand_graph_sh), object_ngbrs_alpha_all, name="expand_graph")
        return expand_graph

    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.num_rois, self.feature_size)
