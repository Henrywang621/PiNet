import tensorflow as tf
# def scheduler(epoch, lr):
#     if epoch % 20 != 0:
#         return lr
#     else:
#         return lr * 0.1
    

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch == 5:
        return lr * 10
    elif epoch > 5 and epoch < 20:
        return lr
    else:
        if lr <= 1e-6:
            return 1e-6
        else:
            return lr * tf.math.exp(-0.2)
        