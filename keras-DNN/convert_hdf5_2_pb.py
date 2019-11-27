import os
import argparse

## --- This convertor runs with tensorflow backend ------------------------ ##
## --- if you get error starting with "Using Theano" ---------------------- ##
## --- or AttributeError: 'module' object has no attribute 'get_session' -- ##
## --- Open /home/$USER_NAME/.keras/keras.json and change the line ------- ##
## --- "backend": "theano" to "backend": "tensorflow" --------------------- ##
## --- and then execute the lines like below  ----------------------------- ##


## --- Example -- ##
## python convert_hdf5_2_pb.py --input <input .hdf5 file name with path> --output <output file name> 


parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="Input Keras model (hdf5)")
parser.add_argument('--output', required=False, type=str, default=None, help="Output Protocol Buffers file")
args = parser.parse_args()

import tensorflow as tf
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras import backend as K
#from common import LoadModel
from keras.models import load_model

K.set_learning_phase(0)
#model = LoadModel(args.input)
model = load_model(args.input)

print(model.inputs[0].name)
print(model.outputs[0].name)
#raise RuntimeError("stop")

#print ("teste", [node.op.name for node in model.outputs])
input_nodes = [model.inputs[0].name] #"main_input"]#  [model.inputs[0].name]
output_nodes = [model.outputs[0].name] #"main_output/Softmax"]#["output_node"]
#input_nodes = ["main_input"]#  [model.inputs[0].name]
#output_nodes = ["main_output/Softmax"]#["output_node"]
#node_wrapper = tf.identity(model.outputs[0], name=output_nodes[0])

with K.get_session() as sess:

    ops = sess.graph.get_operations()
    const_graph = convert_variables_to_constants(sess, sess.graph.as_graph_def(add_shapes=True), [node.op.name for node in model.outputs])
    print ([node.op.name for node in model.outputs])
    # final_graph = const_graph
    final_graph = const_graph


    #for node in model.outputs :
    #    shapes = node.op.attr[model.outputs[0].name]
    #    print (shapes.list.shape[0].dim[0].size)
    #shape = final_graph.get_tensor_by_name(model.outputs[0].name) #.node(0) #final_graph.node(0).attr().at("shape").shape();
    #print ("read input layer shape  " , shape) #shape.dim_size() , graphDef.node(0).name())

if args.output is None:
    input_base = os.path.basename(args.input)
    out_dir = '.'
    out_file = os.path.splitext(input_base)[0] + ".pb"
else:
    out_dir, out_file = os.path.split(args.output)
write_graph(final_graph, out_dir, out_file, as_text=False)
