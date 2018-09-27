import tensorflow as tf


class ClassifyInference(object):

    def __init__(self):

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver = tf.train.import_meta_graph('./classify/model/model.ckpt.meta')
                self.saver.restore(self.sess, tf.train.latest_checkpoint('./classify/model/'))

        self.x = self.graph.get_tensor_by_name("x:0")
        self.logits = self.graph.get_tensor_by_name("logits_eval:0")

    def predict(self, data):
        feed_dict = {self.x: data}
        classification_result = self.sess.run(self.logits, feed_dict)
        with self.sess.as_default():
            with self.graph.as_default():
                return tf.argmax(classification_result, 1).eval(session=self.sess)[0]

    def shutdown(self):
        self.sess.close()
