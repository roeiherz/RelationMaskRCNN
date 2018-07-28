import tensorflow as tf


############################################################
#  Graph Permeation Invariant Model
############################################################
# @todo: delete
class GPILayerModel(object):
    def __init__(self, num_rois, single_rois=None, pairwise_rois=None, rnn_steps=1, gpi_type='FeatureAttention',
                 scope_name='deep_gpi_graph'):
        self.scope_name = scope_name
        # Single ROIs - [num_boxes , 4]
        self.single_rois = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="obj_bb")
        # self.single_rois = tf.squeeze(single_rois, axis=[0])
        # Pairwise ROIs - [num_boxes^2 , 4]
        self.pairwise_rois = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="relation_bb")
        # self.pairwise_rois = tf.squeeze(pairwise_rois, axis=[0])
        self.num_rois = num_rois
        self.gpi_type = gpi_type
        self.rnn_steps = max(1, rnn_steps)
        self.activation_fn = tf.nn.relu

    def run_gpi(self, inputs):
        """
        
        :return: 
        """

        # Crop single features [batch, num_boxes, feature_size]
        entity_features = inputs[0]
        # Crop pairwise features [batch, num_boxes^2, feature_size]
        relation_features = inputs[1]

        # features msg
        for step in range(self.rnn_steps):
            object_feature = self.build_gpi(relation_features=relation_features,
                                            entity_features=entity_features,
                                            scope_name="deep_graph")
            # store the confidence
            # self.out_confidence_entity_lst.append(confidence_entity)
            # self.reuse = True

        return object_feature

    def build_gpi(self, entity_features, relation_features, scope_name="deep_graph"):
        """
        
        :param inputs: 
        :return: 
        """

        # Ignore the batch axis [num_boxes, 1024]
        entity_features_sq = tf.squeeze(entity_features, axis=[0])
        # Ignore the batch axis [num_boxes^2, 1024]
        relation_features_sq = tf.squeeze(relation_features, axis=[0])
        # Reshape [num_boxes, num_boxes, 1024]
        relation_features_sq = tf.reshape(relation_features_sq, [self.num_rois, self.num_rois, 1024])

        # Number of rois
        N = tf.slice(tf.shape(entity_features_sq), [0], [1], name="N")

        # Concat relations in the feature axis - [num_boxes, num_boxes, feature_size * 2]
        relation_features = tf.concat((relation_features_sq, tf.transpose(relation_features_sq, perm=[1, 0, 2])),
                                      axis=2)

        # Concat spatial features - [num_boxes, features_size + 4]
        entity_features = tf.concat((entity_features_sq, self.single_rois), axis=1)
        # Shape [num_boxes, num_boxes , 4]
        shape = tf.concat((N, tf.shape(self.single_rois)), axis=0)
        # Reshape to [num_boxes, num_boxes, 4]
        shaped_relation_bb = tf.reshape(self.pairwise_rois, shape)
        # Concat to feature axis - [num_boxes, num_boxes, feature_size * 2 + 4]
        relation_features = tf.concat((relation_features, shaped_relation_bb), axis=2)

        # Expand object confidence
        extended_confidence_entity_shape = tf.concat((N, tf.shape(entity_features)), 0)
        expand_object_features = tf.add(tf.zeros(extended_confidence_entity_shape), entity_features,
                                        name="expand_object_features")

        # Expand subject confidence
        expand_subject_features = tf.transpose(expand_object_features, perm=[1, 0, 2], name="expand_subject_features")

        # Node Neighbours
        object_ngbrs = [expand_object_features, expand_subject_features, relation_features]

        # Phi
        object_ngbrs_phi = self.nn_model(features=object_ngbrs, layers=[500, 500], out=500, scope_name="nn_phi")

        # Attention mechanism over phi
        if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
            object_ngbrs_scores = self.nn_model(features=object_ngbrs, layers=[500], out=500, scope_name="nn_phi_atten")
            object_ngbrs_weights = tf.nn.softmax(object_ngbrs_scores, dim=1)
            object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(object_ngbrs_phi, object_ngbrs_weights), axis=1)

        elif self.gpi_type == "NeighbourAttention":
            object_ngbrs_scores = self.nn_model(features=object_ngbrs, layers=[500], out=1, scope_name="nn_phi_atten")
            object_ngbrs_weights = tf.nn.softmax(object_ngbrs_scores, dim=1)
            object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(object_ngbrs_phi, object_ngbrs_weights), axis=1)
        else:
            object_ngbrs_phi_all = tf.reduce_sum(object_ngbrs_phi, axis=1) / tf.constant(40.0)

        # Nodes
        object_ngbrs2 = [entity_features, object_ngbrs_phi_all]
        # Alpha
        object_ngbrs2_alpha = self.nn_model(features=object_ngbrs2, layers=[500, 500], out=500, scope_name="nn_phi2")

        # Attention mechanism over alpha
        if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
            object_ngbrs2_scores = self.nn_model(features=object_ngbrs2, layers=[500], out=500,
                                                 scope_name="nn_phi2_atten")
            object_ngbrs2_weights = tf.nn.softmax(object_ngbrs2_scores, dim=0)
            object_ngbrs2_alpha_all = tf.reduce_sum(tf.multiply(object_ngbrs2_alpha, object_ngbrs2_weights), axis=0)
        elif self.gpi_type == "NeighbourAttention":
            object_ngbrs2_scores = self.nn_model(features=object_ngbrs2, layers=[500], out=1,
                                                 scope_name="nn_phi2_atten")
            object_ngbrs2_weights = tf.nn.softmax(object_ngbrs2_scores, dim=0)
            object_ngbrs2_alpha_all = tf.reduce_sum(tf.multiply(object_ngbrs2_alpha, object_ngbrs2_weights), axis=0)
        else:
            object_ngbrs2_alpha_all = tf.reduce_sum(object_ngbrs2_alpha, axis=0) / tf.constant(40.0)

        expand_graph_shape = tf.concat((N, tf.shape(object_ngbrs2_alpha_all)), 0)
        expand_graph = tf.add(tf.zeros(expand_graph_shape), object_ngbrs2_alpha_all)

        ##
        # rho entity (entity prediction)
        # The input is entity features, entity neighbour features and the representation of the graph
        object_all_features = [entity_features, expand_graph]
        obj_delta = self.nn_model(features=object_all_features, layers=[500, 500, 1024], out=2, scope_name="nn_obj")
        out_feature_object = obj_delta
        return [out_feature_object]

    def nn_model(self, features, layers, out, scope_name, last_activation=None):
        """
        simple nn to convert features to confidence
        :param features: list of features tensor
        :param layers: hidden layers
        :param out: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :param last_activation: activation function for the last layer (None means no activation)
        :return: likelihood
        """

        with tf.variable_scope(scope_name) as scopevar:
            index = 0
            h = tf.concat(features, axis=-1)
            for layer in layers:
                scope = str(index)

                # Two 1024 FC layers (implemented with Conv2D for consistency)
                # h = KL.TimeDistributed(KL.Conv2D(layer, (1, 1), padding="valid"), name=scope)(h)
                # h = KL.Activation('relu')(h)

                h = tf.contrib.layers.fully_connected(h, layer, scope=scope, activation_fn=self.activation_fn)
                index += 1

            scope = str(index)
            y = tf.contrib.layers.fully_connected(h, out, scope=scope, activation_fn=last_activation)
        return y
