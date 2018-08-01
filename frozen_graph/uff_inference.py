
import tensorrt as trt
from tensorrt.parsers import uffparser
import tensorrt
import uff

MAX_WORKSPACE = 1 << 30
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

INPUT_W = 28
INPUT_H = 28
OUTPUT_SIZE = 10

MAX_BATCHSIZE = 1
ITERATIONS = 10

#uff_model = uff.from_tensorflow(tf_model, ["fc2/Relu"])

#Convert Tensorflow model to TensorRT model
parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (1024, 2048, 3), 0)
parser.register_output("conv6_cls/BiasAdd")

stream = uff.from_tensorflow_frozen_model("icnet_model.pb" ,["conv6_cls/BiasAdd"])
engine = trt.utils.uff_to_trt_engine(G_LOGGER, stream, parser, MAX_BATCHSIZE, MAX_WORKSPACE)
#engine = trt.utils.uff_file_to_trt_engine(G_LOGGER, "icnet_model.uff", parser, MAX_BATCHSIZE, MAX_WORKSPACE)
#assert(engine)

exit(0)

parser.destroy()
context = engine.create_execution_context()

print("\n| TEST CASE | PREDICTION |")
for i in range(ITERATIONS):
    img, label = lenet5.get_testcase()
    img = img[0]
    label = label[0]
    out = infer(context, img, 1)
    print("|-----------|------------|")
    print("|     " + str(label) + "     |      " + str(np.argmax(out)) + "     |")