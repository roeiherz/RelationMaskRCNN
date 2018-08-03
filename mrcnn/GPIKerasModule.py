import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


############################################################
#  Graph Permeation Invariant Model
############################################################

class GPIKerasModel(object):
    """
    GPI Model - Implementation of GPI module in detection.
    Paper - https://arxiv.org/abs/1802.05451
    Git implementation from https://github.com/shikorab/SceneGrap
    """

    def __init__(self, num_rois, rnn_steps=1, gpi_type='FeatureAttention',
                 feature_size=1024, scope_name='deep_gpi_graph'):
        self.scope_name = scope_name
        # Featre size - default 1024
        self.feature_size = feature_size
        # Single ROIs - [batch, num_boxes , 4]
        self.single_rois = KL.Input(shape=[None, 4], name="obj_bb")
        # Pairwise ROIs - [batch, num_boxes, num_boxes, 4]
        self.pairwise_rois = KL.Input(shape=[None, None, 4], name="relation_bb")
        # Object features - [batch ,num_boxes , 1024]
        self.entity_features = KL.Input(shape=[None, self.feature_size], name="entity_features")
        # Relation features - [batch ,num_boxes^2 , 1024]
        self.relation_features = KL.Input(shape=[None, self.feature_size], name="relation_features")
        self.num_rois = num_rois
        self.gpi_type = gpi_type
        self.rnn_steps = max(1, rnn_steps)
        self.activation_fn = tf.nn.relu

    def build_gpi(self, scope_name="deep_graph_init"):
        """
        
        :param inputs: 
        :return: 
        """
        # Reshape [batch, num_boxes, num_boxes, 1024]
        relation_features_rs = KL.Reshape((self.num_rois, self.num_rois, self.feature_size))(self.relation_features)

        relation_features_pr = KL.Permute(dims=[2, 1, 3])(relation_features_rs)
        # Concat relations in the feature axis - [batch_num, num_boxes, num_boxes, feature_size * 2]
        relation_features = KL.concatenate(inputs=[relation_features_rs, relation_features_pr], axis=3)

        # Concat spatial features - [batch, num_boxes, features_size + 4]
        entity_features = KL.concatenate(inputs=[self.entity_features, self.single_rois], axis=2)
        # Reshape to [batch, num_boxes, features_size + 4]
        entity_features = KL.Reshape((self.num_rois, K.int_shape(entity_features)[2]))(entity_features)
        # Reshape to [batch, num_boxes, num_boxes, 4]
        shaped_relation_bb = KL.Reshape((self.num_rois, self.num_rois, 4))(self.pairwise_rois)
        # Concat to feature axis - [batch, num_boxes, num_boxes, feature_size * 2 + 4]
        relation_features = KL.concatenate(inputs=[relation_features, shaped_relation_bb], axis=3)

        m = KM.Model([self.single_rois, self.pairwise_rois, self.entity_features, self.relation_features],
                     [entity_features, relation_features])

        return m

    def expand_object_subject(self, entity_features, relation_features, scope_name="deep_graph_expand"):
        """
        
        :param inputs: 
        :return: 
        """
        # Expand object confidence
        # [num_boxes, num_boxes, features_size + 4]
        extended_confidence_entity_shape = (self.num_rois, self.num_rois, entity_features.shape[2])
        # [num_boxes, num_boxes, features_size + 4]
        # expand_object_features = KL.add([K.zeros(extended_confidence_entity_shape), K.squeeze(entity_features, 0)],
        #                                 name="expand_object_features")
        expand_object_features = tf.add(tf.zeros(extended_confidence_entity_shape), tf.squeeze(entity_features, 0),
                                        name="expand_object_features")
        # Add batch - [batch, num_boxes, num_boxes, features_size + 4]
        # expand_object_features = KL.Lambda(lambda x: K.expand_dims(x, axis=0))(expand_object_features)

        # Expand subject confidence
        # expand_subject_features = KL.Permute(dims=[2, 1, 3], name="expand_subject_features")(expand_object_features)
        expand_subject_features = tf.transpose(expand_object_features, perm=[1, 0, 2], name="expand_subject_features")

        # Node Neighbours
        object_ngbrs = [expand_object_features, expand_subject_features, tf.squeeze(relation_features, 0)]
        # object_ngbrs_tensor = KL.Lambda(lambda x: K.concatenate(x, axis=-1))(object_ngbrs)
        object_ngbrs_tensor = tf.concat(object_ngbrs, axis=-1)

        return object_ngbrs_tensor

    def phi(self, entity_features, object_ngbrs_tensor, scope_name="deep_graph_phi"):
        """
        
        :param inputs: 
        :return: 
        """

        # Phi
        object_ngbrs_phi = self.mlp_model(features=object_ngbrs_tensor, size=object_ngbrs_tensor.shape[2],
                                          layers=[500, 500], out=500, scope_name="nn_phi")
        # Attention mechanism over phi
        if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
            object_ngbrs_scores = self.mlp_model(features=object_ngbrs_tensor, size=object_ngbrs_tensor.shape[2],
                                                 layers=[500], out=500, scope_name="nn_phi_atten")
            object_ngbrs_weights = tf.nn.softmax(object_ngbrs_scores, axis=1)
            object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(object_ngbrs_phi, object_ngbrs_weights), axis=1)

        elif self.gpi_type == "NeighbourAttention":
            object_ngbrs_scores = self.mlp_model(features=object_ngbrs_tensor, size=object_ngbrs_tensor.shape[2],
                                                 layers=[500], out=1, scope_name="nn_phi_atten")
            object_ngbrs_weights = tf.nn.softmax(object_ngbrs_scores, axis=1)
            object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(object_ngbrs_phi, object_ngbrs_weights), axis=1)
        else:
            object_ngbrs_phi_all = tf.reduce_sum(object_ngbrs_phi, axis=1) / tf.constant(40.0)

        # Nodes
        object_ngbrs2 = [tf.squeeze(entity_features, 0), object_ngbrs_phi_all]
        # object_ngbrs2_tensor = KL.Lambda(lambda x: K.concatenate(x, axis=-1))(object_ngbrs2)
        object_ngbrs2_tensor = tf.concat(object_ngbrs2, axis=-1)
        return object_ngbrs2_tensor

    def alpha(self, object_ngbrs2_tensor, scope_name="deep_graph_alpha"):
        """
        
        :param inputs: 
        :return: 
        """

        # Alpha
        object_ngbrs2_alpha = self.mlp_model(features=object_ngbrs2_tensor, size=object_ngbrs2_tensor.shape[1],
                                             layers=[500, 500], out=500, scope_name="nn_phi2")

        # Attention mechanism over alpha
        if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
            object_ngbrs2_scores = self.mlp_model(features=object_ngbrs2_tensor, size=object_ngbrs2_tensor.shape[1],
                                                  layers=[500], out=500, scope_name="nn_phi2_atten")
            object_ngbrs2_weights = tf.nn.softmax(object_ngbrs2_scores, axis=0)
            object_ngbrs2_alpha_all = tf.reduce_sum(tf.multiply(object_ngbrs2_alpha, object_ngbrs2_weights), axis=0)
        elif self.gpi_type == "NeighbourAttention":
            object_ngbrs2_scores = self.mlp_model(features=object_ngbrs2_tensor, size=object_ngbrs2_tensor.shape[1],
                                                  layers=[500], out=1, scope_name="nn_phi2_atten")
            object_ngbrs2_weights = tf.nn.softmax(object_ngbrs2_scores, axis=0)
            object_ngbrs2_alpha_all = tf.reduce_sum(tf.multiply(object_ngbrs2_alpha, object_ngbrs2_weights), axis=0)
        else:
            object_ngbrs2_alpha_all = tf.reduce_sum(object_ngbrs2_alpha, axis=0) / tf.constant(40.0)

        # [num_boxes, 500]
        # expand_graph = KL.add([K.zeros(object_ngbrs2_alpha_all.shape), object_ngbrs2_alpha_all], name="expand_graph")
        expand_graph_sh = (self.num_rois,) + K.int_shape(object_ngbrs2_alpha_all)
        expand_graph = tf.add(tf.zeros(expand_graph_sh), object_ngbrs2_alpha_all, name="expand_graph")
        # Add batch - [batch, num_boxes, 500]
        # expand_graph = KL.Lambda(lambda x: K.expand_dims(x, axis=0))(expand_graph)
        return expand_graph

    def rho(self, entity_features, object_alpha_tensor, scope_name="deep_graph_rho"):
        """
        
        :param inputs: 
        :return: 
        """

        # rho entity (entity prediction)
        # The input is entity features, entity neighbour features and the representation of the graph
        object_all_features = [tf.squeeze(entity_features, 0), object_alpha_tensor]
        # object_all_features_tensor = KL.Lambda(lambda x: K.concatenate(x, axis=-1))(object_all_features)
        object_all_features_tensor = tf.concat(object_all_features, axis=-1)

        out_feature_object = self.mlp_model(features=object_all_features_tensor,
                                            size=object_all_features_tensor.shape[1],
                                            layers=[500, 500], out=1024, scope_name="nn_obj")
        # out_feature_object = KL.Lambda(lambda x: x, name="out_feature_object")(out_feature_object)
        # out_feature_object = tf.expand_dims(out_feature_object, axis=0)
        return out_feature_object

    def mlp_model(self, features, size, layers, out, scope_name='', last_activation=None):
        """
        simple nn to convert features to confidence
        :param features: keras tensor after concat
        :param layers: hidden layers
        :param out: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :param last_activation: activation function for the last layer (None means no activation)
        :return: likelihood
        """

        with tf.variable_scope(scope_name) as scopevar:
            # input_layer = KL.Input(shape=tuple(features.get_shape().as_list()[1:]))
            # features.shape[1:]
            h = features
            for i, layer in enumerate(layers):
                # h = KL.Dense(layer, activation='relu', name='{}'.format(i))(h)
                h = tf.contrib.layers.fully_connected(h, layer, scope='{}'.format(i),
                                                      activation_fn=self.activation_fn)

            # out_layer = KL.Dense(out, name='out')(h)
            y = tf.contrib.layers.fully_connected(h, out, scope='{}'.format(i+1), activation_fn=last_activation)

            # Create Model
            # m = KM.Model(inputs=[input_layer], outputs=[out_layer])

        # return m(features)
        return y
