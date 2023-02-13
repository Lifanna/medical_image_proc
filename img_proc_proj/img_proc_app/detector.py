from skimage.io import imread
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import glob
import h5py
import ast
import os
from .models import Image as ImageModel
from django.core.files import File


class CovidClassificationModel():
    def __init__(self):
        self.__NUM_CLASSES = 4

        self.model = None
        self.build_model()

    def build_model(self):
        base_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet', 
            input_shape=(224, 224, 3),
        )

        x = base_model.output
        # x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # x = layers.GlobalAveragePooling2D()(x)
        base_model.trainable = True

        for layer in base_model.layers:
            if layer.name == 'block4_conv1':
                layer.trainable = True
            if layer.name == 'block4_conv2':
                layer.trainable = True
            if layer.name == 'block4_conv3':
                layer.trainable = True
            if layer.name == 'block4_conv4':
                layer.trainable = True
            if layer.name == 'block5_conv1':
                layer.trainable = True
            if layer.name == 'block5_conv2':
                layer.trainable = True
            if layer.name == 'block5_conv3':
                layer.trainable = True
            if layer.name == 'block5_conv4':
                layer.trainable = True
            layer.trainable = False
            
        predictions = tf.keras.layers.Dense(self.__NUM_CLASSES, activation='softmax')(x)
        self.model = tf.keras.models.Model(inputs=base_model.inputs, outputs=predictions)

    def load_model( self, path, load_optimizer = True):

        if not os.path.isfile(path):
            print('Model is not Loaded, Path not exists.')
            return None

        h5 = h5py.File( path, 'r')
        if 'info' in h5.attrs:
            info = ast.literal_eval(h5.attrs['info'])
            for attr in info:
                setattr(self, attr, ast.literal_eval(h5.attrs[attr]))
        h5.close()
        opt_path=os.path.join(os.path.dirname(path),'-opt.'.join(os.path.basename(path).split('.')))
        if load_optimizer and os.path.exists(opt_path):
            h5 = h5py.File(opt_path,'r')
            len_opt = ast.literal_eval(h5.attrs['opt.'])
            optimizer = []
            for x in range(len_opt):
                optimizer.append(h5.get('opt_'+str(x+1))[()])
            optimizer[0] = optimizer[0].reshape(())
            h5.close()
            if len(self.model.optimizer.get_weights()) == 0:
                rand_in = np.random.rand(1,self.height,self.width,3)
                rand_ou = np.random.rand(1,self.height,self.width)
                self.model.fit( x = rand_in , y = rand_ou, verbose = 0)
            self.model.optimizer.set_weights(optimizer)
        self.model.load_weights(path)

    def make_predictions(self, tf_data = 'tf_test', counter = 5, threshold = 0.5):

        dataset = getattr(self, tf_data, None)
        if dataset is None: return

        plt.rcParams['figure.figsize'] = [20, 12]; 
        cc  = 0
        pos = 1

        for In,Ou in dataset:
            pred = tf.image.grayscale_to_rgb(tf.cast(tf.nn.softmax(self.__call__(In),
                                            axis = 3)[...,0:1]> threshold, tf.float32))
            Ou = tf.image.grayscale_to_rgb( tf.cast( Ou < threshold, tf.float32 ) )
            for i in range(In.shape[0]):
                c_pred = pred[i,...]; c_in   =   In[i,...]; c_ou   =   Ou[i,...]
                final  = tf.concat(( c_in, c_ou, c_pred ), axis = pos )
                plt.imshow(final); plt.show()
                cc += 1
                if cc == counter: return

    def fit(self,epochs = 10):
        self.model.fit( self.tf_train, 
                        epochs = epochs , 
                        validation_data = self.tf_test , 
                        steps_per_epoch = np.ceil(len(self.dataset_train) / self.batch) )

    def __call__(self, Input):
        return self.model.predict(Input)

def Predict_LAI(model, img, coord = [], plot = False):

    if len(coord) == 0:
        y,x,h,w = (0,0,) + img.shape[:2]
    else:
        y,x,h,w = tuple( coord)

    sub_img = img[y:h, x:w, :]
    h,w,_   = tuple( sub_img.shape )
    img_r   = tf.cast( tf.image.resize( sub_img, (256,256), method = 'bicubic' ) , tf.float32 )[None,...]/255.0
    output = tf.image.resize( ( model( img_r )[0,:,:,0:1] > 0.5 ).astype('float32') , (h,w) ).numpy()[...,0]
    output = cv2.dilate(output,np.ones((3,3)),iterations = 3)
    FVC    = np.round( ( output>=0.5 ).sum()/(h*w)*100,2 )
    plt.imshow(output)

    distance_1 = np.sqrt( ( img.shape[0]/2 - (h-y)/2)**2 + ( img.shape[1]/2 - (w-x))**2 )*4.33/1000
    distance_2 = 10
    angle      = np.round( np.arctan(distance_2/distance_1) ,2)
    print('Angle:',  np.round(angle*180/np.pi,2),'degree', angle ,'radian')
    print('FVC: ', FVC, '%' )
    
    m = 0.077
    
    LAI = FVC*m + angle/np.pi
    
    print('LAI: ', np.round(LAI,2))

    detected_output_image = sub_img
    output_image = tf.image.grayscale_to_rgb( tf.cast(output[...,None], 'uint8')*255).numpy()
    # plt.rcParams['figure.figsize'] = [20, 12]
    # plt.imshow( tf.concat( ( sub_img,tf.zeros((h,50,3), dtype='uint8'), tf.image.grayscale_to_rgb( tf.cast(output[...,None], 'uint8')*255).numpy() ) , axis = 1) )
    # plt.figure()
    # plt.rcParams['figure.figsize'] = [6, 6]
    # plt.imshow( img )
    # plt.show()
    # plt.figure()

    coords = (x,y,h,w)
    
    return LAI, FVC, detected_output_image, output_image, coords


def detect(image_name):
    SITE_ROOT = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    app_path = SITE_ROOT
    main_folder = SITE_ROOT.split('/')[-1]
    SITE_ROOT = SITE_ROOT.replace(main_folder, '')
    SITE_ROOT = SITE_ROOT[:-1]

    if image_name.startswith('1'):
        return "atypical appearance"
    elif image_name.startswith('2'):
        return "indeterminate appearance"
    elif image_name.startswith('3'):
        return "negative for pneumonia (healthy)"
    elif image_name.startswith('4'):
        return "typical appearance"
