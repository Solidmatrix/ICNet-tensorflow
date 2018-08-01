"""
convert_to_uff.py

Main script for doing uff conversions from
different frameworks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import uff


def process_cmdline_args():
    """
    Helper function for processing commandline arguments
    """
    # general arguments applicable to all subparser
    # adding parent parser for usability
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-o", "--output",
        help="""Name of output uff file.""")

    parent_parser.add_argument(
        '-l', '--list-nodes', action='store_true',
        help="""Show list of nodes contained in input file.""")

    parent_parser.add_argument(
        '-t', '--text', action='store_true',
        help="""Write a text version of the output in addition to the
        binary.""")

    parent_parser.add_argument(
        '-q', '--quiet', action='store_true',
        help="""Disable log messages.""")

    parser = argparse.ArgumentParser(
        description="""Utility to convert TensorFlow models to
            Unified Framework Format (UFF).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(
        title="Available Converters.",
        description="tensorflow",
        help="Valid commands to do conversion.")

    # add tensorflow subparser
    parser_tf = subparsers.add_parser("tensorflow", parents=[parent_parser])

    parser_tf.add_argument(
        "--input-file", required=True,
        help="""Path to input model. Can be a frozen GraphDef or a
        SavedModel.""")

    parser_tf.add_argument(
        "-O", "--output-node", action='append',
        help="""Name of a node to mark as an output of the model.""")

    parser_tf.add_argument(
        '-I', '--input-node', action='append',
        help="""Name of a node to replace with an input to the model.
        Must be specified as: "name_or_index,new_name,dtype,dim1,dim2,..."
        """)

    parser_tf.add_argument(
        "-p", "--preprocessor",
        help="""The TF preprocessing file to run before handling the graph.""")

    parser_tf.set_defaults(func=uff.from_tensorflow_frozen_model)

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    if args.input_node is None:
        args.input_node = []
    if args.input_file.endswith('/'):
        args.input_file = args.input_file[:-1]
    if args.output is None:
        args.output = args.input_file + '.uff'
    else:
        if not args.output.endswith(".uff"):
            args.output = args.output + '.uff'

    return args, unknown_args


def main():
    args, _ = process_cmdline_args()
    if not args.quiet:
        print("Loading", args.input_file)
    args.func(
        args.input_file,
        output_nodes=args.output_node,
        preprocessor=args.preprocessor,
        input_node=args.input_node,
        quiet=args.quiet,
        text=args.text,
        list_nodes=args.list_nodes,
        output_filename=args.output
    )


if __name__ == '__main__':
    main()
