#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = "./quantize_graph/icnet_quantize_graph.pb"
    #sess.graph.add_to_collection("input", mnist.test.images)

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        input = sess.graph.get_tensor_by_name("conv2_1_1x1_proj:0")
        print input
        #output = sess.graph.get_tensor_by_name("output:0")
        #print output

        #pred = sess.run(output, {input: todo})
        #print "pred: ", pred