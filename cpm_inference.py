import importlib
import numpy as np
import os

import tensorflow as tf

from config import FLAGS
from utils import tracking_module


class CPMInference(object):

    def __init__(self):
        cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)

        self.model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                         heatmap_size=FLAGS.heatmap_size,
                                         stages=FLAGS.cpm_stages,
                                         joints=FLAGS.num_of_joints,
                                         img_type=FLAGS.color_channel,
                                         is_training=False)
        saver = tf.train.Saver()

        self.output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)

        device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
        sess_config = tf.ConfigProto(device_count=device_count)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)

        model_path_suffix = os.path.join(FLAGS.network_def,
                                         'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                         'joints_{}'.format(FLAGS.num_of_joints),
                                         'stages_{}'.format(FLAGS.cpm_stages),
                                         'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate,
                                                                          FLAGS.lr_decay_step)
                                         )
        model_save_dir = os.path.join('models',
                                      'weights',
                                      model_path_suffix)
        print('Load model from [{}]'.format(os.path.join(model_save_dir, FLAGS.model_path)))
        saver.restore(self.sess, 'models/weights/cpm_hand')

        # Check weights
        for variable in tf.global_variables():
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable.name.split(':0')[0])
                print(variable.name, np.mean(self.sess.run(var)))

    def predict(self, img):
        return self.sess.run([self.output_node], feed_dict={self.model.input_images: img})

    def shutdown(self):
        self.sess.close()
