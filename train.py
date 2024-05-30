import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from PiNet import *
from loss import *
from utils import scheduler
from tensorflow.keras.metrics import binary_crossentropy


npy_path =  '/home/henrywang/PiNet/Data/npyfiles/'
checkpoint_path = 'Checkpoints1/'
log_path = 'log1/'

width = 352
height = 352
batch_size = 24
epoch = 45
lr = 1e-5 
load_weight = 0

X_train = np.load(npy_path + 'X_train_full.npy')
X_val = np.load(npy_path + 'VX_Validation1.npy')
y_train = np.load(npy_path + 'y_train_full.npy')
y_val = np.load(npy_path + 'VY_Validation1.npy')
edge_label = np.load(npy_path + 'Detail_train_full.npy')
edge_val = np.load(npy_path + 'VDetail_Validation1.npy')
model = PiNet(width, height)
# model.summary()

model_checkpoint = ModelCheckpoint('Checkpoints1/PiNet.{loss:.3f}.hdf5', monitor='final_loss',verbose=1, period=1, save_weights_only=True, save_best_only=True)
# lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=2, mode = 'min', min_delta=0.0001, min_lr=1e-8)

# if load_weight:
#     model.load_weights('Checkpoints1/PiNet.0.461.hdf5',by_name=False)

lr_decay = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
tensorboard = TensorBoard(log_dir=log_path)
callback_list = [lr_decay, model_checkpoint, tensorboard]
adam = Adam(learning_rate = lr)
lossWeights = {"edge1": 1, "edge2": 1, "o1": 1, 'o2':1, 'o3':1, 'final':4}
model.compile(loss = {'edge1': binary_crossentropy,
                      'edge2': binary_crossentropy,
                      'o1': Sharpenning_Loss,
                      'o2': Sharpenning_Loss,
                      'o3': Sharpenning_Loss,
                      'final': Sharpenning_Loss,
                      },
                      loss_weights=lossWeights,
                      metrics={'final': ['mae', 'acc']},
                      optimizer=adam)

model.fit(X_train, {'edge1':edge_label, 'edge2':edge_label, 'o1':y_train, 'o2':y_train, 'o3':y_train, 'final':y_train}, 
          validation_data = (X_val, {'edge1':edge_val, 'edge2':edge_val, 'o1':y_val, 'o2':y_val, 'o3':y_val, 'final':y_val}), 
          shuffle=True, batch_size=batch_size, epochs=epoch, callbacks=callback_list, verbose=2)