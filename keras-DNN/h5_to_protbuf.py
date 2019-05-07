# Convert .h5 model from keras to .pb for tensorflow.
# .pb stands for "protbuf" contains the graph definition as well as the model weights.
# All you need to run trained model.

import os
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
K.set_session
from keras.models import Model,load_model
from keras.layers import Dense, Input

def load_graph_test(test_frozen_graph):
    print 'Loading outputs . . . '
    # Load protobuf file from disk
    # parse it to retireve unserialized graph_def
    with tf.gfile.GFile(test_frozen_graph, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())

    # Import graph_def into a new graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')

    return graph

usage = 'usage: %prog [options]'
parser = argparse.ArgumentParser(usage)
parser.add_argument('-i', '--input_h5', dest='input_h5', help='Input Keras model file.h5 to convert to Tensorflow graph', default=None, type=str)
parser.add_argument('-o', '--output_pb', dest='output_pb', help='Output Tensorflow protocol buffers file.pb', default=None, type=str)
args = parser.parse_args()

input_h5 = args.input_h5
output_pb = args.output_pb
output_human_readable = output_pb + '.ascii'

K.set_learning_phase(0)
model = load_model(input_h5)

print 'outputs'
for index in xrange(len(model.outputs)):
    print 'node.op.name = ', model.outputs[index].op.name
    print 'identity = ', tf.identity(model.outputs[index], name=model.outputs[index].op.name)

sess = K.get_session()

output_names = [out_node.op.name for out_node in model.outputs]

tf.train.write_graph(sess.graph.as_graph_def(), '.', output_human_readable, as_text=True)

input_graph_def = sess.graph.as_graph_def()

predicted_names = output_names
#predicted_names += [v.op.name for v in tf.global_variables()]

print "predicted_names = ", predicted_names

# Freeze graph
constant_graph = convert_variables_to_constants(sess, input_graph_def, predicted_names)

from tensorflow.python.framework import graph_io
graph_io.write_graph(constant_graph, '.', output_pb, as_text=False)

loaded_graph_test = load_graph_test(output_pb)
for op in loaded_graph_test.get_operations():
    print 'loaded_graph_test: ', op.name
