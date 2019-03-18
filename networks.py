# -*- coding: utf-8 -*-
 
import tensorflow as tf 

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.*intersection + smooth)/ (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return score

def jaccard_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f) 
    union        = tf.reduce_sum(tf.math.maximum(y_true_f, y_pred_f))
    score = (intersection + smooth)/ (union + smooth)
    return score

def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def deconv(in_shape, cout=1, loss="binary_crossentropy", opt='adadelta', lr=1., compile_it=True): 
    input_img = tf.keras.layers.Input(shape=in_shape)
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c1_1')(input_img)  
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c1_2')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c1_3')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c1_4', strides=2)(x)     
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c2_1')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c2_2')(x)    
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c2_3')(x) 
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c2_4', strides=2)(x)    
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_1')(x)  
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_2')(x)   
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_3')(x)   
    x = tf.keras.layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same', name='d1', strides=2)(x)    
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c4_1')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c4_2')(x)     
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c4_3')(x)     
    x = tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', name='d2', strides=2)(x)    
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c5_1')(x)   
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c5_2')(x)    
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c5_3')(x)      
    x = tf.keras.layers.Conv2D(cout, (3,3), activation='sigmoid', padding='same', name='cout')(x) 
    
    model = tf.keras.Model(inputs=[input_img], outputs=[x], name='model_cae_threes')
    if not compile_it:
        return model

    if loss == 'bce_dice':
        loss = bce_dice_loss
    elif loss == 'jaccard':
        loss = jaccard_loss
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model

def NoDownSample_net(in_shape, cout=1, loss="binary_crossentropy",  final_activ='sigmoid',
                    opt='adadelta', lr=1., compile_it=True):
    def residual_unit(x, filters, size, name=''):  
        f1, f2 = filters  
        c = tf.keras.layers.Conv2D(f1, (size,size), padding='same', name=name+'_c1')(x)  
        c = tf.keras.layers.BatchNormalization(name=name+'_bn1')(c)   
        c = tf.keras.layers.Activation('relu', name=name+'_r1')(c)

        c = tf.keras.layers.Conv2D(f2, (1,1), padding='same', name=name+'_c2')(x)  
        c = tf.keras.layers.BatchNormalization(name=name+'_bn2')(c)   
        c = tf.keras.layers.Activation('relu', name=name+'_r2')(c) 

        c = tf.keras.layers.Add(name=name+'_ad')([c, x])  
        c = tf.keras.layers.Activation('relu', name=name+'_r3')(c)
        return c

    input_img = tf.keras.layers.Input(shape=in_shape, name='input')    
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c1')(input_img)    
    x = residual_unit(x, (64, 32), 3, name='c1_1')       
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c2')(x)   
    x = residual_unit(x, (128, 64), 3, name='c2_1')     
    x = residual_unit(x, (128, 64), 3, name='c2_2')   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c3')(x) 
    x = residual_unit(x, (64, 32), 3, name='c3_1')   
    x = tf.keras.layers.Conv2D(cout, (3,3), activation=final_activ, padding='same', name='cout')(x) 
    model = tf.keras.models.Model(input_img, x, name='NoDownSample_net')
    if not compile_it:
        return model
    
    if loss == 'bce_dice':
        loss = bce_dice_loss 
    elif loss == 'jaccard':
        loss = jaccard_loss
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model
 

def mini_unet(in_shape, cout=1, loss="binary_crossentropy", final_activ='sigmoid',
             opt='adadelta', lr=1., compile_it=True): 
    input_img = tf.keras.layers.Input(shape=in_shape)
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c1_1')(input_img)  
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c1_2')(x)   
    c1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c1_3')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c1_4', strides=2)(c1)     
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c2_1')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c2_2')(x)    
    c2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c2_3')(x) 
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c2_4', strides=2)(c2)    
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_1')(x)  
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_2')(x)   
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_3')(x)   
    x = tf.keras.layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same', name='d1', strides=2)(x)   
    x = tf.keras.layers.Concatenate(-1, name='m1')([x, c2]) 
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c4_1')(x)   
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c4_2')(x)     
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c4_3')(x)     
    x = tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', name='d2', strides=2)(x)    
    x = tf.keras.layers.Concatenate(-1, name='m2')([x, c1]) 
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c5_1')(x)   
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c5_2')(x)    
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c5_3')(x)      
    x = tf.keras.layers.Conv2D(cout, (3,3), activation=final_activ, padding='same', name='cout')(x) 

    model = tf.keras.Model(inputs=[input_img], outputs=[x], name='mini_unet')
    if not compile_it:
        return model
    
    if loss == 'bce_dice':
        loss = bce_dice_loss 
    elif loss == 'jaccard':
        loss = jaccard_loss
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model

def light_deconv(in_shape, cout, loss="binary_crossentropy", opt='adadelta', lr=1., compile_it=True): 
    input_img = tf.keras.layers.Input(shape=in_shape)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c1_1')(input_img)  
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c1_2')(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c1_4', strides=2)(x)     
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c2_1')(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c2_4', strides=2)(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c3_1')(x)    
    x = tf.keras.layers.Conv2DTranspose(8, (3,3), activation='relu', padding='same', name='d1', strides=2)(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c4_1')(x)       
    x = tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', name='d2', strides=2)(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c5_1')(x)         
    x = tf.keras.layers.Conv2D(cout, (3,3), activation='sigmoid', padding='same', name='cout')(x)  
    model = tf.keras.Model(inputs=[input_img], outputs=[x], name='model_cae_threes')
    if not compile_it:
        return model
    
    if loss == 'bce_dice':
        loss = bce_dice_loss 
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model


def light_unet(in_shape, cout, loss="binary_crossentropy", opt='adadelta', lr=1., compile_it=True): 
    input_img = tf.keras.layers.Input(shape=in_shape)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c1_1')(input_img)  
    x1 = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c1_2')(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c1_4', strides=2)(x1)     
    x2 = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c2_1')(x)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c2_4', strides=2)(x2)    
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c3_1')(x)    
    x = tf.keras.layers.Conv2DTranspose(8, (3,3), activation='relu', padding='same', name='d1', strides=2)(x) 
    x = tf.keras.layers.Concatenate()([x, x2])   
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c4_1')(x)       
    x = tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', name='d2', strides=2)(x)    
    x = tf.keras.layers.Concatenate()([x, x1])   
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', name='c5_1')(x)         
    x = tf.keras.layers.Conv2D(cout, (3,3), activation='sigmoid', padding='same', name='cout')(x)  
    model = tf.keras.Model(inputs=[input_img], outputs=[x], name='model_cae_threes')
    if not compile_it:
        return model
    
    if loss == 'bce_dice':
        loss = bce_dice_loss 
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model

def unet(in_shape, cout=1, loss="binary_crossentropy", opt='adadelta', lr=1.):
    def conv_block(input_tensor, num_filters):
        encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('relu')(encoder)
        encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)
        decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)
        decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)
        return decoder 
    inputs = tf.keras.layers.Input(shape=in_shape) 
    encoder0_pool, encoder0 = encoder_block(inputs, 32) 
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)  
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) 
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) 
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) 
    center = conv_block(encoder4_pool, 1024) 
    decoder4 = decoder_block(center, encoder4, 512) 
    decoder3 = decoder_block(decoder4, encoder3, 256) 
    decoder2 = decoder_block(decoder3, encoder2, 128) 
    decoder1 = decoder_block(decoder2, encoder1, 64) 
    decoder0 = decoder_block(decoder1, encoder0, 32) 
    outputs = tf.keras.layers.Conv2D(cout, (1, 1), activation='sigmoid')(decoder0) 

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    if loss == 'bce_dice':
        loss = bce_dice_loss
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])   
    return model

def denseNet(in_shape, final_activ='sigmoid', opt='adadelta', loss='binary_crossentropy',
             cout=1, lr=1., compile_it=True, nb_dense_block=2, growth_rate=2, nb_filter=8, reduction=0.5,
             dropout_rate=0.0, weight_decay=1e-4, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5



    def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor 
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)
        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4  
        x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x) 
        x = tf.keras.layers.Activation('relu', name=relu_name_base+'_x1')(x)
        x = tf.keras.layers.Conv2D(inter_channel, 1, padding='same', name=conv_name_base+'_x1', use_bias=False)(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        # 3x3 Convolution
        x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x) 
        x = tf.keras.layers.Activation('relu', name=relu_name_base+'_x2')(x) 
        x = tf.keras.layers.Conv2D(nb_filter, 3, padding='same', name=conv_name_base+'_x2', use_bias=False)(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x
    
    def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage) 
        x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x) 
        x = tf.keras.layers.Activation('relu', name=relu_name_base)(x)
        x = tf.keras.layers.Conv2D(int(nb_filter * compression), 1, padding='same', name=conv_name_base, use_bias=False)(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)
        return x
    
    def up_transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'ups' + str(stage) 
        x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x) 
        x = tf.keras.layers.Activation('relu', name=relu_name_base)(x)
        x = tf.keras.layers.Conv2D(int(nb_filter * compression), 1, padding='same', name=conv_name_base, use_bias=False)(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)  
        x = tf.keras.layers.UpSampling2D(size=(2,2), name=pool_name_base)(x) 
        return x


    def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''
        concat_feat = x
        for i in range(nb_layers):
            branch = i+1
            x = conv_block(concat_feat, stage, branch, nb_filter, dropout_rate, weight_decay)
            concat_feat = tf.keras.layers.Concatenate(axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))([concat_feat, x])
            if grow_nb_filters:
                nb_filter += growth_rate
        return concat_feat, nb_filter
    

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    concat_axis = 3
    img_input = tf.keras.layers.Input(shape=in_shape, name='data') 
    
    # From architecture for ImageNet (Table 1 in the paper)
    #nb_filter = 64
    nb_layers = [6,12,6] # For DenseNet-121

    # Initial convolution 
    x = tf.keras.layers.Conv2D(nb_filter, 7, padding='same',  name='conv1', use_bias=False)(img_input)
    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x) 
    x = tf.keras.layers.Activation('relu', name='relu1')(x)
    
    # Add dense blocks
    for block_idx in range(nb_dense_block):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, 
                                   growth_rate, dropout_rate=dropout_rate, 
                                   weight_decay=weight_decay) 
        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, 
                             dropout_rate=dropout_rate, weight_decay=weight_decay) 
        nb_filter = int(nb_filter * compression)

     
    for block_idx in range(nb_dense_block ):    
        up_stage = block_idx + stage + 2
        x, nb_filter = dense_block(x, up_stage, nb_layers[block_idx], nb_filter, 
                                   growth_rate, dropout_rate=dropout_rate, 
                                   weight_decay=weight_decay) 
        # Add up_transition_block
        x = up_transition_block(x, up_stage, nb_filter, compression=compression, 
                                dropout_rate=dropout_rate, weight_decay=weight_decay) 
        nb_filter = int(nb_filter * compression)
 
    final_stage = up_stage + 1
    x = tf.keras.layers.BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x) 
    x = tf.keras.layers.Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = tf.keras.layers.Conv2D(cout, 3, padding="same", activation=final_activ)(x) 
   
    model = tf.keras.models.Model(img_input, x, name='densenet')
    if not compile_it:
        return model
    
    if loss == 'bce_dice':
        loss = bce_dice_loss
    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    else:
        opt = tf.keras.optimizers.Adadelta(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy']) 

    #model.summary()
    return model 


def get_net(net_name, in_shape, cout=1, opt='adadelta', final_activ='sigmoid',
            loss='binary_crossentropy', lr=1., compile_it=True):
    
    if net_name=='deconv':
        return deconv(in_shape, cout=cout, opt=opt, loss=loss, lr=lr, compile_it=compile_it)

    if net_name=='ndsnet':
        return NoDownSample_net(in_shape, cout=cout, opt=opt, loss=loss, lr=lr, compile_it=compile_it)
    
    if net_name=='unet':
        return unet(in_shape, cout=cout, opt=opt, loss=loss, lr=lr)

    if net_name=='munet':
        return mini_unet(in_shape, cout=cout, opt=opt, loss=loss, final_activ=final_activ, lr=lr, compile_it=compile_it)
    
    if net_name=='dense':
        return denseNet(in_shape, cout=cout, opt=opt, loss=loss, lr=lr, compile_it=compile_it)

    if net_name=='light':
        return light_deconv(in_shape, cout=cout, opt=opt, loss=loss, lr=lr, compile_it=compile_it)
 

    raise ValueError('Unknown net_name, expected (). Got {}'.format(net_name))

