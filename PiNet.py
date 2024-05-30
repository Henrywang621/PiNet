import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB3, EfficientNetB4

class SDConv2d_SSC(tf.keras.layers.Layer):
    """
    implement Smoothed Dilated Convolution (Separable and Shared Convolution)
    """
    def __init__(self, num_o, kernel_size, dilation_factor, name, biased=False, **kwargs):
        super().__init__(name = name, **kwargs)
        self.num_o = num_o
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor
        self.biased = biased
        self.fix_w_size = dilation_factor * 2 - 1
        self.mask = np.zeros([self.fix_w_size, self.fix_w_size, 1, 1, 1], dtype=np.float32)
        self.mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1

    def build(self, input_shape):
        num_x = input_shape[-1]
        self.fix_w = self.add_weight(
            shape=(self.fix_w_size, self.fix_w_size, 1, 1, 1),
            initializer=tf.constant_initializer(self.mask),
            name='fix_w',
            trainable=True,
        )
        self.w = self.add_weight(
            shape=(self.kernel_size, self.kernel_size, num_x, self.num_o),
            initializer="uniform",
            name='w',
            trainable=True,
        )
        if self.biased:
            self.b = self.add_weight(
                shape=self.num_o,
                initializer="uniform",
                name='b',
                trainable=True,
            )

    def call(self, inputs):
        o = tf.expand_dims(inputs, -1)
        o = tf.nn.conv3d(o, self.fix_w, strides=[1, 1, 1, 1, 1], padding='SAME')
        o = tf.squeeze(o, -1)
        o = tf.nn.atrous_conv2d(o, self.w, self.dilation_factor, padding='SAME')
        if self.biased:
            o = tf.nn.bias_add(o, self.b)
        return o
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_o': self.num_o,
            'kernel_size': self.kernel_size,
            'dilation_factor': self.dilation_factor,
            'biased': self.biased,
            'fix_w_size': self.fix_w_size,
            'mask': self.mask,
        })
        return config


def SDConv_Bn_ReLU(output_channel_number,
                   kernel_size,
                   dilation_factor,
                   name,
                   batchnorm=True,
                   activation_layer='ReLU'):
    def layer(input_tensor):
        x = SDConv2d_SSC(output_channel_number, kernel_size,
                         dilation_factor, name)(input_tensor)
        if batchnorm:
            x = BatchNormalization(epsilon = 1e-6)(x)

        if activation_layer == 'ReLU':
            x = Activation('relu')(x)
        elif activation_layer == 'Sigmoid':
            x = Activation('sigmoid')(x)
        elif activation_layer == 'Softmax':
            x = Activation('softmax')(x)
        else:
            raise ValueError("The name of the activation layer must be one of the following options: ReLU, Sigmoid, Softmax.")
        return x
    return layer


def Conv_Bn_ReLU(num_filters,
                 kernel_size,
                 name,
                 batchnorm=True,
                 activation_layer='ReLU',
                 strides=(1, 1),
                 dilation_rate=(1, 1),
                 padding='same'):
    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size, name=name,
                   padding=padding, kernel_initializer='TruncatedNormal',
                   strides=strides, dilation_rate=dilation_rate)(input_tensor)
        if batchnorm:
            x = BatchNormalization(epsilon = 1e-6)(x)

        if activation_layer == 'ReLU':
            x = Activation('relu')(x)
        elif activation_layer == 'Sigmoid':
            x = Activation('sigmoid')(x)
        elif activation_layer == 'Softmax':
            x = Activation('softmax')(x)
        else:
            raise ValueError("The name of the activation layer must be one of the following options: ReLU, Sigmoid, Softmax.")
        return x
    return layer

def PiNet(img_width, img_height, backbone_name='ResNet50'):

    input_shape = (img_width, img_height, 3)

    # inputs = Input(input_shape)
    if backbone_name == 'EfficientNet3B':
        backbone = EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet')

        #Get side-output
        Layer_name1 = backbone.layers[28].name
        Layer_name2 = backbone.layers[72].name
        Layer_name3 = backbone.layers[116].name
        Layer_name4 = backbone.layers[263].name
        Layer_name5 = backbone.layers[-1].name
        side_output1 = backbone.get_layer(Layer_name1).output
        side_output2 = backbone.get_layer(Layer_name2).output
        side_output3 = backbone.get_layer(Layer_name3).output
        side_output4 = backbone.get_layer(Layer_name4).output
        side_output5 = backbone.get_layer(Layer_name5).output
        
    elif backbone_name == 'EfficientNet4B':
        backbone = EfficientNetB4(input_shape=input_shape, include_top=False, weights='imagenet')

        #Get side-output
        Layer_name1 = backbone.layers[28].name
        Layer_name2 = backbone.layers[87].name
        Layer_name3 = backbone.layers[146].name
        Layer_name4 = backbone.layers[323].name
        Layer_name5 = backbone.layers[-1].name
        side_output1 = backbone.get_layer(Layer_name1).output
        side_output2 = backbone.get_layer(Layer_name2).output
        side_output3 = backbone.get_layer(Layer_name3).output
        side_output4 = backbone.get_layer(Layer_name4).output
        side_output5 = backbone.get_layer(Layer_name5).output

    elif backbone_name == 'VGG16':
        backbone = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

        #Get side-output
        Layer_name1 = backbone.layers[3].name
        Layer_name2 = backbone.layers[6].name
        Layer_name3 = backbone.layers[10].name
        Layer_name4 = backbone.layers[14].name
        Layer_name5 = backbone.layers[-1].name
        side_output1 = backbone.get_layer(Layer_name1).output
        side_output2 = backbone.get_layer(Layer_name2).output
        side_output3 = backbone.get_layer(Layer_name3).output
        side_output4 = backbone.get_layer(Layer_name4).output
        side_output5 = backbone.get_layer(Layer_name5).output

    elif backbone_name == 'ResNet50':
        backbone = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

        #Get side-output
        Layer_name1 = backbone.layers[4].name
        Layer_name2 = backbone.layers[38].name
        Layer_name3 = backbone.layers[80].name
        Layer_name4 = backbone.layers[142].name
        Layer_name5 = backbone.layers[-1].name
        side_output1 = backbone.get_layer(Layer_name1).output
        side_output2 = backbone.get_layer(Layer_name2).output
        side_output3 = backbone.get_layer(Layer_name3).output
        side_output4 = backbone.get_layer(Layer_name4).output
        side_output5 = backbone.get_layer(Layer_name5).output

    else:
        raise ValueError("The type of the backbone must be VGG16, ResNet50, Efficient3B or Efficient4B!!!")


    # HLPIB3 --- get features from CA module
    conv1_side_o1h3 = Conv_Bn_ReLU(128, (1, 1), 'conv1_side_output1h3')(side_output5)

    # Smoothed Feature Interaction Module (SFIM)
    conv1_SFIM3_dr1 = Conv_Bn_ReLU(128, (1, 1), 'conv1_SFIM3_r1')(conv1_side_o1h3)
    conv1_SFIM3_dr3 = SDConv_Bn_ReLU(128, kernel_size=3, dilation_factor=2, name='conv1_SFIM3_r3')(conv1_side_o1h3)

    conv1_SFIM3_dr21 = Add()([conv1_SFIM3_dr1, conv1_side_o1h3, conv1_SFIM3_dr3])
    conv1_SFIM3_dr2 = Conv_Bn_ReLU(128, (3, 3), 'conv1_SFIM3_r2')(conv1_SFIM3_dr21)

    conv1_SFIM3_dr12 = Add()([conv1_SFIM3_dr1, conv1_side_o1h3, conv1_SFIM3_dr2])
    conv1_SFIM3_dr22 = Add()([conv1_SFIM3_dr2, conv1_SFIM3_dr21])
    conv1_SFIM3_dr32 = Add()([conv1_SFIM3_dr3, conv1_side_o1h3, conv1_SFIM3_dr2])


    conv2_SFIM3_dr13 = Conv_Bn_ReLU(96, (1, 1), 'conv2_SFIM3_r13')(conv1_SFIM3_dr12)
    conv2_SFIM3_dr23 = Conv_Bn_ReLU(96, (1, 1), 'conv2_SFIM3_r23')(conv1_SFIM3_dr22)
    conv2_SFIM3_dr33 = Conv_Bn_ReLU(96, (1, 1), 'conv2_SFIM3_r33')(conv1_SFIM3_dr32)
    conv2_SFIM3_output = Add()([conv2_SFIM3_dr13, conv2_SFIM3_dr23, conv2_SFIM3_dr33])

    # CA
    CA3_gp = GlobalAveragePooling2D()(conv2_SFIM3_output)
    CA3_fc1 = Dense(32, activation="relu", name='CA3_fc1')(CA3_gp)
    CA3_fc2 = Dense(96, activation="sigmoid", name='CA3_fc2')(CA3_fc1)

    updated_features3 = Multiply()([CA3_fc2, conv2_SFIM3_output])
    updated_features3_up = UpSampling2D(size=(8, 8))(updated_features3)


    # Salient Edge Feature Extractor (SEFE)
    conv1_side_o1e = Conv_Bn_ReLU(96, (3, 3), 'conv1_side_output1e')(side_output1)
    conv1_side_o5e = Conv_Bn_ReLU(96, (3, 3), 'conv1_side_output5e')(updated_features3_up)

    conv2_side_o5e = UpSampling2D(size=(2, 2))(conv1_side_o5e)
    fusion_mf = Add()([conv1_side_o1e, conv2_side_o5e])
    conv2_fusion_mf = Conv_Bn_ReLU(96, (3, 3), 'conv2_fusion_mf')(fusion_mf)
    conv3_fusion_mf = Conv_Bn_ReLU(96, (3, 3), 'conv3_fusion_mf')(conv2_fusion_mf)
    conv4_fusion_mf = Conv_Bn_ReLU(96, (3, 3), 'conv4_fusion_mf')(conv3_fusion_mf)
    edge_extract = Conv2D(1, (1, 1), padding='same', kernel_initializer='TruncatedNormal', name='conv5_fusion_mf')(conv4_fusion_mf)
    edge_extract = Activation('sigmoid', name='edge1')(edge_extract)


    # Low-Level Progressive Improvement Branch (LLPIB) --- Containing Saliency Prediction (SP) and Edge Prediction (EP) sub-branches
    conv1_side_o2l = Conv_Bn_ReLU(96, (3, 3), 'conv1_side_output1l')(side_output2)
    conv1_side_o5l = Conv_Bn_ReLU(96, (3, 3), 'conv1_side_output5l')(updated_features3_up)
    conv2_side_o2l = Conv_Bn_ReLU(96, (3, 3), 'conv2_side_output1l')(conv1_side_o2l)

    conv2_fusion = Add()([conv2_side_o2l, conv1_side_o5l])

    # Smoothed Feature Attention Module (SFAM)
    conv1_SFAM_dr1 = Conv_Bn_ReLU(32, (3, 3), 'conv1_SFAM_rate1')(conv2_fusion)
    conv1_SFAM_dr2 = SDConv_Bn_ReLU(32, kernel_size=3, dilation_factor=2, name='conv1_SFAM_rate2')(conv2_fusion)
    conv1_SFAM_dr3 = SDConv_Bn_ReLU(32, kernel_size=3, dilation_factor=3, name='conv1_SFAM_rate3')(conv2_fusion)

    # Three summarized feature maps used to generate three attention maps
    attention_dr1 = Conv_Bn_ReLU(1, (1, 1), 'conv3_attention_dr1', activation_layer='Sigmoid')(conv1_SFAM_dr1)
    attention_dr2 = Conv_Bn_ReLU(1, (1, 1), 'conv3_attention_dr2', activation_layer='Sigmoid')(conv1_SFAM_dr2)
    attention_dr3 = Conv_Bn_ReLU(1, (1, 1), 'conv3_attention_dr3', activation_layer='Sigmoid')(conv1_SFAM_dr3)

    # Get the attention maps
    attention1 = Flatten()(attention_dr1)
    attention2 = Flatten()(attention_dr2)
    attention3 = Flatten()(attention_dr3)
    attention1 = Softmax()(attention1)
    attention2 = Softmax()(attention2)
    attention3 = Softmax()(attention3)

    # Apply the attention maps to the corresponding features
    attention1 = RepeatVector(32)(attention1)
    attention2 = RepeatVector(32)(attention2)
    attention3 = RepeatVector(32)(attention3)
    attention1 = Permute((2, 1))(attention1)
    attention2 = Permute((2, 1))(attention2)
    attention3 = Permute((2, 1))(attention3)
    attention1 = Reshape((88, 88, 32), name='attention_1')(attention1)
    attention2 = Reshape((88, 88, 32), name='attention_2')(attention2)
    attention3 = Reshape((88, 88, 32), name='attention_3')(attention3)
    attention1 = Multiply()([attention1, conv1_SFAM_dr1])
    attention2 = Multiply()([attention2, conv1_SFAM_dr2])
    attention3 = Multiply()([attention3, conv1_SFAM_dr3])
    weighted_features1 = Add()([attention1, conv1_SFAM_dr1])
    weighted_features2 = Add()([attention2, conv1_SFAM_dr2])
    weighted_features3 = Add()([attention3, conv1_SFAM_dr3])

    SFAM_output = concatenate([weighted_features1, weighted_features2, weighted_features3])

    # the EP branch of the 2th layer
    conv3_LLPIM_edge = Conv_Bn_ReLU(96, (3, 3), 'conv2_LLPIB_ep')(conv2_fusion)
    SFAM_edge_fusion = Add()([SFAM_output, conv3_LLPIM_edge])
    # the EP branch of the 3th layer
    conv4_LLPIM_edge = Conv_Bn_ReLU(96, (3, 3), 'conv3_LLPIB_ep')(SFAM_edge_fusion)
    # Output the predicted edge of the input
    LLPIM_edge_output = Conv2D(1, (1, 1), padding='same', kernel_initializer='TruncatedNormal', name='LLPIM_edge_output')(conv4_LLPIM_edge)
    LLPIM_edge_output = Activation('sigmoid', name='edge2a')(LLPIM_edge_output)
    LLPIM_edge_output = UpSampling2D(size=(2, 2), name='edge2')(LLPIM_edge_output)


    #HLPIB3 --- get saliency features
    # Aggregate edge features and updated features
    updated_features3 = Add()([updated_features3_up, conv4_LLPIM_edge])
    conv2_h3 = Conv_Bn_ReLU(96, (3, 3), 'conv2_h3')(updated_features3)
    updated_features3_up = UpSampling2D(size=(2, 2))(conv2_h3)
    updated_features3 = Add()([updated_features3_up, conv4_fusion_mf])

    conv3_h3 = Conv_Bn_ReLU(96, (3, 3), 'conv3_h3')(updated_features3)
    Features5 = Conv_Bn_ReLU(96, (3, 3), 'conv4_h3')(conv3_h3)
    HLPIB3_output = Conv2D(1, (1, 1), padding='same', kernel_initializer='TruncatedNormal', name='h3_output')(Features5)
    HLPIB3_output = Activation('sigmoid', name='h3_oa')(HLPIB3_output)
    HLPIB3_output = UpSampling2D(size=(2, 2), name='o3')(HLPIB3_output)


    # Hight-Level Progressive Improvement Branch2 (HLPIB2)
    conv1_side_o1h2 = Conv_Bn_ReLU(128, (1, 1), 'conv1_side_output1h2')(side_output4)

    # Smoothed Feature Interaction Module (SFIM)
    conv1_SFIM2_dr1 = Conv_Bn_ReLU(128, (3, 3), 'conv1_HLPIB2_r1')(conv1_side_o1h2)
    conv1_SFIM2_dr3 = SDConv_Bn_ReLU(128, kernel_size=3, dilation_factor=3, name='conv1_HLPIB2_r3')(conv1_side_o1h2)

    conv1_SFIM2_dr21 = Add()([conv1_SFIM2_dr1, conv1_side_o1h2, conv1_SFIM2_dr3])
    conv1_SFIM2_dr2 = SDConv_Bn_ReLU(128, kernel_size=3, dilation_factor=2, name='conv1_HLPIB2_r2')(conv1_SFIM2_dr21)

    conv1_SFIM2_dr12 = Add()([conv1_SFIM2_dr1, conv1_side_o1h2, conv1_SFIM2_dr2])
    conv1_SFIM2_dr22 = Add()([conv1_SFIM2_dr2, conv1_SFIM2_dr21])
    conv1_SFIM2_dr32 = Add()([conv1_SFIM2_dr3, conv1_side_o1h2, conv1_SFIM2_dr2])

    conv2_SFIM2_dr13 = Conv_Bn_ReLU(96, (1, 1), 'conv2_SFIM2_r13')(conv1_SFIM2_dr12)
    conv2_SFIM2_dr23 = Conv_Bn_ReLU(96, (1, 1), 'conv2_SFIM2_r23')(conv1_SFIM2_dr22)
    conv2_SFIM2_dr33 = Conv_Bn_ReLU(96, (1, 1), 'conv2_SFIM2_r33')(conv1_SFIM2_dr32)
    conv2_SFIM2_output = Add()([conv2_SFIM2_dr13, conv2_SFIM2_dr23, conv2_SFIM2_dr33])

    # CA
    CA2_gp = GlobalAveragePooling2D()(conv2_SFIM2_output)
    CA2_fc1 = Dense(32, activation="relu", name='CA2_fc1')(CA2_gp)
    CA2_fc2 = Dense(96, activation="sigmoid", name='CA2_fc2')(CA2_fc1)

    updated_features2 = Multiply()([CA2_fc2, conv2_SFIM2_output])


    # Aggregate edge features and updated features
    updated_features2_up = UpSampling2D(size=(4, 4))(updated_features2)
    updated_features2 = Add()([updated_features2_up, conv4_LLPIM_edge])
    conv2_h2 = Conv_Bn_ReLU(96, (3, 3), 'conv2_h2')(updated_features2)
    updated_features2_up = UpSampling2D(size=(2, 2))(conv2_h2)
    updated_features2 = Add()([updated_features2_up, conv4_fusion_mf, Features5])

    conv3_h2 = Conv_Bn_ReLU(96, (3, 3), 'conv3_h2')(updated_features2)
    Features4 = Conv_Bn_ReLU(96, (3, 3), 'conv4_h2')(conv3_h2)
    HLPIM2_output = Conv2D(1, (1, 1), padding='same', kernel_initializer='TruncatedNormal', name='h2_output')(Features4)
    HLPIM2_output = Activation('sigmoid', name='h2_oa')(HLPIM2_output)
    HLPIM2_output = UpSampling2D(size=(2, 2), name='o2')(HLPIM2_output)


    # Transition Branch1 (TB)
    conv1_side_o1h1 = Conv_Bn_ReLU(96, (1, 1), 'conv1_side_output1h1')(side_output3)

    # Smoothed Feature Attention Module (SFAM)
    conv1_SFAMTB_dr1 = Conv_Bn_ReLU(32, (3, 3), 'conv1_SFAM_TB_rate1')(conv1_side_o1h1)
    conv1_SFAMTB_dr2 = SDConv_Bn_ReLU(32, kernel_size=3, dilation_factor=2, name='conv1_SFAM_TB_rate2')(conv1_side_o1h1)
    conv1_SFAMTB_dr3 = SDConv_Bn_ReLU(32, kernel_size=3, dilation_factor=3, name='conv1_SFAM_TB_rate3')(conv1_side_o1h1)

    # Three summarized feature maps used to generate three attention maps
    attention_TB_dr1 = Conv_Bn_ReLU(1, (1, 1), 'conv3_attention_TB_dr1', activation_layer='Sigmoid')(conv1_SFAMTB_dr1)
    attention_TB_dr2 = Conv_Bn_ReLU(1, (1, 1), 'conv3_attention_TB_dr2', activation_layer='Sigmoid')(conv1_SFAMTB_dr2)
    attention_TB_dr3 = Conv_Bn_ReLU(1, (1, 1), 'conv3_attention_TB_dr3', activation_layer='Sigmoid')(conv1_SFAMTB_dr3)

    # Get the attention maps
    attention1_TB = Flatten()(attention_TB_dr1)
    attention2_TB = Flatten()(attention_TB_dr2)
    attention3_TB = Flatten()(attention_TB_dr3)
    attention1_TB = Softmax()(attention1_TB)
    attention2_TB = Softmax()(attention2_TB)
    attention3_TB = Softmax()(attention3_TB)

    # Apply the attention maps to the corresponding features
    attention1_TB = RepeatVector(32)(attention1_TB)
    attention2_TB = RepeatVector(32)(attention2_TB)
    attention3_TB = RepeatVector(32)(attention3_TB)
    attention1_TB = Permute((2, 1))(attention1_TB)
    attention2_TB = Permute((2, 1))(attention2_TB)
    attention3_TB = Permute((2, 1))(attention3_TB)
    attention1_TB = Reshape((44, 44, 32), name='TB_attention_1')(attention1_TB)
    attention2_TB = Reshape((44, 44, 32), name='TB_attention_2')(attention2_TB)
    attention3_TB = Reshape((44, 44, 32), name='TB_attention_3')(attention3_TB)
    attention1_TB = Multiply()([attention1_TB, conv1_SFAMTB_dr1])
    attention2_TB = Multiply()([attention2_TB, conv1_SFAMTB_dr2])
    attention3_TB = Multiply()([attention3_TB, conv1_SFAMTB_dr3])
    weighted_features1_TB = Add()([attention1_TB, conv1_SFAMTB_dr1])
    weighted_features2_TB = Add()([attention2_TB, conv1_SFAMTB_dr2])
    weighted_features3_TB = Add()([attention3_TB, conv1_SFAMTB_dr3])

    SFAM_TB_output = concatenate([weighted_features1_TB, weighted_features2_TB, weighted_features3_TB])

    # Aggregate edge features and updated features
    updated_features1_up = UpSampling2D(size=(2, 2))(SFAM_TB_output)
    updated_features1 = Add()([updated_features1_up, conv4_LLPIM_edge])
    conv2_TB = Conv_Bn_ReLU(96, (3, 3), 'conv2_TB')(updated_features1)
    updated_features1_up = UpSampling2D(size=(2, 2))(conv2_TB)
    updated_features1 = Add()([updated_features1_up, conv4_fusion_mf, Features4])

    conv3_h1 = Conv_Bn_ReLU(96, (3, 3), 'conv3_TB')(updated_features1)
    Features3 = Conv_Bn_ReLU(96, (3, 3), 'conv4_TB')(conv3_h1)
    TB_output = Conv2D(1, (1, 1), padding='same', kernel_initializer='TruncatedNormal', name='TB_output')(Features3)
    TB_output = Activation('sigmoid', name='TB_oa')(TB_output)
    TB_output = UpSampling2D(size=(2, 2), name='o1')(TB_output)


    # LLPIM -- finish the saliency prediction branch

    #get Features2
    LLPIM_sm_output = Add()([SFAM_output, conv4_LLPIM_edge])
    LLPIM_sm_output = Conv_Bn_ReLU(96, (3, 3), 'LLPIM_output1')(LLPIM_sm_output)
    Features2 = Conv_Bn_ReLU(96, (3, 3), 'LLPIM_output2')(LLPIM_sm_output)

    Features2_up = UpSampling2D(size=(2, 2))(Features2)
    updated_Features2 = Add()([Features2_up, Features3])
    Features1 = Conv_Bn_ReLU(64, (3, 3), 'conv_Features1')(updated_Features2)
    Features1_up = UpSampling2D(size=(2, 2))(Features1)
    final_saliency_map = Conv2D(1, (1, 1), padding='same', kernel_initializer='TruncatedNormal', name='final_output')(Features1_up)
    final_saliency_map = Activation('sigmoid', name='final')(final_saliency_map)

    PiNet = Model(inputs=backbone.input, outputs=[edge_extract, LLPIM_edge_output, TB_output, HLPIM2_output, HLPIB3_output, final_saliency_map])
    return PiNet