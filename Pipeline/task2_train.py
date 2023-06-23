import os,random
import cv2 as cv
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
from imagemorph import elastic_morphing

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

def Read_character_data(path,letters):
    print('Reading images...')
    valid=[]
    test=[]
    train=[]
    classes = os.listdir(path) 
    print('\t Classes:', classes)
    if '.DS_Store' in classes:
        classes.remove('.DS_Store') 
        print('\t Removed .DS_stored')
    for subdir in classes: 
        if '.DS_Store'== subdir:
            continue
        current_path = os.path.join(path, subdir)
        files=os.listdir(current_path)
        num=len(files)
        counter = 0
        for i in range(len(files)):
            if files[i].endswith('.jpg') or files[i].endswith('.png') or files[i].endswith('.jpeg') or files[i].endswith('.pgm'):
                im = cv.imread(os.path.join(current_path, files[i]))
                im = img_pre_process(im)
                im=cv.resize(im,(50,50), interpolation=cv.INTER_LANCZOS4)
                idx= letters.index(subdir)
                lb = to_categorical(idx, 27)
                if i<int(num*0.7):
                    train.append([im,lb])
                elif (i<int(num*0.9)):
                    test.append([im,lb])
                else:
                    valid.append([im,lb])
            else:
                print("file does not end with jpg,png,jpeg,pgm")
    print("* Done reading characters.")
    return train,valid,test

letters=['Alef', 'Ayin', 'Bet','Dalet','Gimel','He','Het','Kaf','Kaf-final', 'Lamed','Mem', 'Mem-medial', 'Nun-final',
           'Nun-medial', 'Pe', 'Pe-final', 'Qof','Resh','Samekh','Shin','Taw','Tet','Tsadi-final','Tsadi-medial',
           'Waw','Yod','Zayin']

def add_noise(img):
    h,w=img.shape[0:2]
    for i in range(random.randint(0,3)):
        lf_tp=(random.randint(0,h-6),random.randint(0,w-6))
        intv=random.randint(2,5) 
        rt_bt=(lf_tp[0]+intv,lf_tp[1]+intv)
        if img[lf_tp[0],lf_tp[1],1]<100:
            img=cv.rectangle(img,lf_tp,rt_bt,(220,220,220),-1)
        else:
            img=cv.rectangle(img,lf_tp,rt_bt,(50,50,50),-1)
    return img
            
def model_build():
    BaseCnn = MobileNetV2(input_shape=(50, 50, 3), alpha=1.0,
                               include_top=False,weights='imagenet')
    model=models.Sequential()
    model.add(BaseCnn)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(27,activation = 'softmax'))
    return model


def img_pre_process(img):
    s = max(img.shape[0:2])
    f = np.ones((s,s,3),np.uint8)*255
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img    
    return f
    
def data_generator(data,st,ed,batch_size,aug = None): 
    nowinx = st
    while True:
        im_array = []
        lb_array = []
        for i in range(batch_size):
            res=data[nowinx][0]
            if aug!=None:
                amp, sigma = 2, 9 # coustomized?
                h, w, _= res.shape
                res = elastic_morphing(res, amp, sigma, h, w)
                res=add_noise(res)
            tmp_im_array = res
            tmp_im_array = tmp_im_array[np.newaxis,:,:,:] 
            lb=data[nowinx][1]

            if len(im_array) == 0:
                im_array = tmp_im_array
                lb_array.append(lb)
            else:
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) 
                lb_array.append(lb) 
            if (nowinx==ed) & (aug!=None): 
                random.shuffle(train)
                print("------train set shuffled-------")
            nowinx = st if nowinx==ed else nowinx+1 
            
        lb_array = np.array(lb_array)
        if (aug is not None) and (random.random()>0.3) :
            im_array_old = im_array.copy()
            new_array=im_array
            new_array = next(aug.flow(x=new_array,y=None,batch_size = batch_size, shuffle=False, 
                                      sample_weight=None, seed=None, save_to_dir=None, save_prefix='', 
                                      save_format='png', subset=None))
            im_array=new_array

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
            cv.imwrite("asasd.png",im_array[i])'''
            
        yield(im_array,lb_array)

        
        
aug = ImageDataGenerator(rotation_range = 20,zoom_range = 0.1,width_shift_range = 0.15,
                         height_shift_range = 0.15,shear_range = 20,horizontal_flip =False,vertical_flip =False,
                         fill_mode = "constant",cval=255)

def main(data_path='./monkbrill2_aug/'):
    # parameter
    #data_path='./monkbrill2_aug/'
    batch_size_train=80
    batch_size_vali=30
    
    # main
    solve_cudnn_error()
    train,valid,test=Read_character_data(data_path,letters)
    print('train',len(train),'valid',len(valid),'test',len(test),",Number Check:",
          (len(train)+len(valid)+len(test))/300==27)
    random.shuffle (train)

    train_gen = data_generator(train,0,len(train)-1,batch_size_train,aug) 
    valid_gen = data_generator(valid,0,len(valid)-1,batch_size_vali,None) 


    #model=model_build()
    model=load_model("model/CR_adam_monk1.hdf5")
    model.compile(optimizer = Adam(lr=0.00001),
                  loss = "categorical_crossentropy", metrics = ['accuracy'])

    model.summary()

    reducelrOnplateau=ReduceLROnPlateau(monitor='loss',factor=0.333,patience=3,verbose=1,min_lr=0.000001)

    model_file_path="./model/CR_adam_monk2.hdf5"
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='val_loss', 
        verbose=1,
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
    )
    earlystopping=EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

    tensorboard = TensorBoard(log_dir='./logs/CR_adam_monk2', update_freq='batch')
    '''   '''
    his = model.fit_generator(
        generator = train_gen, 
        callbacks=[tensorboard,model_checkpointer,reducelrOnplateau,earlystopping],  # 
        steps_per_epoch = int(len(train)/batch_size_train), 
        validation_data = valid_gen, 
        validation_steps =int(len(valid)/batch_size_vali), 
        epochs = 100,
        initial_epoch=29
    )

if __name__ == '__main__':
    data_path='./monkbrill2_aug/'
    main(data_path)

