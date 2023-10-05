import tf2onnx
import tensorflow as tf
import tensorflow.keras.backend as kb

import tensorrt as trt

def custom_loss(y_actual, y_pred):
    mask = kb.greater(y_actual, 0)
    mask = tf.cast(mask, tf.float32)
    custom_loss = tf.math.reduce_sum(
        kb.square(mask*(y_actual-y_pred)))/tf.math.reduce_sum(mask)
    return custom_loss

model = tf.keras.models.load_model('1_9_both_trt_test.h5', custom_objects={'custom_loss': custom_loss})

# Load your model


# Convert the model to ONNX format
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model)

# Save the ONNX model
with open("model3.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())




TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def build_engine_onnx(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        

        

        # For TensorRT 7+
        config = builder.create_builder_config()

        # Check if the platform supports FP16 mode
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)  # Set FP16 mode
        else:
            print("Warning: This platform does not have fast FP16 support.")
            return None
        
       
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB


        
        # Optimization profile for dynamic input shapes
        profile = builder.create_optimization_profile()
        
        # Assuming the input tensor name is "input_tensor" and its dynamic shape is [batch_size, channels, height, width]
        # Here's an example where we assume the batch size can vary between 1 and 32, and we optimize for a batch size of 16:
        min_shape = (1, 80, 8)
        opt_shape = (1, 80, 8)
        max_shape = (1, 80, 8)
        
        profile.set_shape("my_input_layer", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Load the ONNX model and parse it to populate the TensorRT network
        with open(model_path, 'rb') as model:
            parser.parse(model.read())
        
        # Check for errors after parsing
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        
        # Build the engine
        engine = builder.build_serialized_network(network, config)
        if not engine:
            print("Failed to build the engine!")
            return None

        return engine

# Convert ONNX model to TensorRT engine
engine = build_engine_onnx("model3.onnx")

# If engine was successfully built, serialize it to a .trt file
if engine:
    with open("model3.trt", "wb") as f:
        f.write(engine)
        # f.write(engine.buffer)
else:
    print("Engine not built successfully!")



