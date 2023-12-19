# Compute gradient of the output of a model <model> with respect to its input <inp>
# If the output of the model is a vector, <output_index> choses which element of output to use
def gradient_input(model,inp,output_index=0):
    inp_tensor = tf.convert_to_tensor(inp, dtype=tf.float32) 
    with tf.GradientTape() as t:
        t.watch(inp_tensor) # enable gradient recording w.r.t. to this tensor
        output = tf.squeeze(model(inp_tensor)) # forward inference
        if output.ndim>0:
            my_output = output[output_index] # pick right element from output 
        else:
            my_output = output
    gradient = t.gradient(my_output,inp_tensor) # get gradient @my_output/@input
    gradient = np.squeeze(np.array(gradient)) # convert from tensor to numpy array
    return gradient