import os,random
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw
from imagemorph import elastic_morphing
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import config
from tensorflow.keras import models,layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def solve_cudnn_error():
    gpus = config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                config.experimental.set_memory_growth(gpu, True)
            logical_gpus = config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

#Returns a grayscale image based on specified label of img_size
char_map = {'Alef':')','Ayin':'(','Bet':'b','Dalet':'d','Gimel':'g','He':'x','Het':'h', 
            'Kaf':'k','Kaf-final':'\\','Lamed':'l', 'Mem':'{','Mem-medial':'m','Nun-final':'}', 
            'Nun-medial':'n','Pe':'p','Pe-final':'v', 'Qof':'q','Resh':'r','Samekh':'s', 
            'Shin' : '$','Taw':'t','Tet':'+','Tsadi-final':'j','Tsadi-medial':'c', 'Waw':'w', 
            'Yod':'y','Zayin':'z'}

letters=['Alef', 'Ayin', 'Bet','Dalet','Gimel','He','Het','Kaf','Kaf-final', 'Lamed','Mem', 'Mem-medial', 'Nun-final',
           'Nun-medial', 'Pe', 'Pe-final', 'Qof','Resh','Samekh','Shin','Taw','Tet','Tsadi-final','Tsadi-medial',
           'Waw','Yod','Zayin']

def create_image(label, img_size):
    if (label not in char_map):
        raise KeyError('Unknown label!')
        
    font = ImageFont.truetype('./Habbakuk.TTF', 42, encoding="unic")
    #Create blank image and create a draw interface
    img = Image.new('RGB', img_size, (255,255,255))    
    draw = ImageDraw.Draw(img)

    #Get size of the font and draw the token in the center of the blank image
    w,h = font.getsize(char_map[label])
    draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), char_map[label], fill =0,font = font)

    return img

def pre_model_build():
    BaseCnn = MobileNetV2(input_shape=(50, 50, 3), alpha=1.0,
                               include_top=False,weights='imagenet')
    model=models.Sequential()
    model.add(BaseCnn)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(27,activation = 'softmax'))
    return model


def data_generator(batch_size,aug = None): 
    while True:
        im_array = []
        lb_array = []
        for i in range(batch_size):
            lb=random.randint(0,26)
            img = create_image(letters[lb], (50, 50))
            img=np.array(img)
            amp, sigma = 2, 9 # coustomized?
            h, w, _= img.shape
            res = elastic_morphing(img, amp, sigma, h, w)
            tmp_im_array = np.array(res)
            tmp_im_array = tmp_im_array[np.newaxis,:,:,:] 
            
            ''' 
            while(1):
                cv.imshow('or',img)
                cv.imshow('res',res)
                if cv.waitKey(1) == ord('0'):
                    break
            cv.destroyAllWindows()'''
            
            if len(im_array) == 0:
                im_array = tmp_im_array
                lb_array.append(lb)
            else:
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) 
                lb_array.append(lb) 
                
        lb_array = to_categorical(lb_array, 27)
        if (aug is not None) and (random.random()>0.2):
            im_array_old = im_array.copy()
            new_array = im_array
            new_array = next(aug.flow(x=new_array,y=None,batch_size = batch_size, shuffle=False, 
                                      sample_weight=None, seed=None, save_to_dir=None, save_prefix='', 
                                      save_format='png', subset=None))
            im_array = new_array#.astype(np.uint8)

        if np.max(im_array)>2:
            im_array=im_array/255
        '''    
        for i in range(batch_size):
            print(im_array[i])
            cv.imwrite("asasd.png",im_array[i])
            print("lb  %s"%letters[np.argmax(lb_array[i])])
            while(1):
                cv.imshow('im_array_old',im_array_old[i])
                cv.imshow('new',im_array[i])
                if cv.waitKey(1) == ord('0'):
                    break
            cv.destroyAllWindows()
            #cv.imwrite("asasd.png",im_array[i])  ''' 
        yield(im_array,lb_array)

        
aug = ImageDataGenerator(rotation_range = 20,zoom_range = 0.1,width_shift_range = 0.15,
                         height_shift_range = 0.15,shear_range = 20,horizontal_flip =False,vertical_flip =False,
                         fill_mode = "constant",cval=255)

def main():
    # parameter
    batch_size_train=50
    batch_size_vali=20
    
    # main
    solve_cudnn_error()

    train_gen = data_generator(batch_size_train,aug) 
    valid_gen = data_generator(batch_size_vali,None) 


    model=pre_model_build()
    #model=load_model('./model/CR_adam_01.hdf5')
    model.compile(optimizer = Adam(lr=0.001),
                  loss = "categorical_crossentropy", metrics = ['accuracy'])

    model.summary()

    reducelrOnplateau=ReduceLROnPlateau(monitor='loss',factor=0.333,patience=3,verbose=1,min_lr=0.000001)

    model_file_path="./model/CR_adam_pre_1.hdf5"
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='loss', 
        verbose=1,
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
    )

    tensorboard = TensorBoard(log_dir='./logs/CR_adam_pre_1', update_freq='batch')
    earlystopping=EarlyStopping(monitor='loss', min_delta=0, patience=8, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

    '''   '''
    his = model.fit_generator(
        generator = train_gen, 
        callbacks=[tensorboard,model_checkpointer,reducelrOnplateau,earlystopping],  # 
        steps_per_epoch = 10, 
        validation_data = valid_gen, 
        validation_steps =5, 
        epochs = 200
    )


if __name__ == '__main__':
    main()

