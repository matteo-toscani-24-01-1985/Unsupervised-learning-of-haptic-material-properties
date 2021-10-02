import numpy as np
import os
import time
import scipy.io
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D 
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger
from numpy.random import seed
seed(1)

from scipy import signal
################################################################################
# COMMONLY SET PARAMETERS
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"



epochs = 50
batch_size = 32 #

################################################################################

tic = time.time()

#### DEFINE LOW PASS FILTER
fs=3200
order=10
cutOff=800
normalized_cutoff_freq = 2 * cutOff / fs
numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)



###############################################
# READ DATA
import os
img_dir = './DATA/' 

all_frames=os.listdir(img_dir)
all_frames = sorted(all_frames)
#desired_im_sz = (  1024,1) 
desired_im_sz = (  256,1) 
#desired_im_sz = (  3072,1)
Nsubsamples=125;
TotalSamples = np.shape(all_frames)[0]*Nsubsamples*2
Begin=32000-desired_im_sz[0]*Nsubsamples
X = np.zeros(( TotalSamples,) + desired_im_sz); # allocate space
counter=0;
for s in range(np.shape(all_frames)[0]):
    filename=all_frames[s]
    im=np.loadtxt(os.path.join(img_dir, filename), dtype='float', comments='#', delimiter=',')
                  
    ## LOW PASS- following weber
    im = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, im)
              
    #im_min=np.min(im[Begin:])
    #im_max=np.max(im[Begin:]-im_min)
    #im=(im-im_min)/im_max
    for ss in range(Nsubsamples):
        b=Begin+ss*desired_im_sz[0]
        end = b+desired_im_sz[0]
        SubSample= im[b:end]
        ###### notmalize (0 1)
        SubSample = SubSample-min(SubSample)
        SubSample = SubSample / max(SubSample)
        X[counter]=np.expand_dims(SubSample,1)
        counter = counter+1
        #
        X[counter]=np.flipud(np.expand_dims(SubSample,1))
        counter = counter+1
 
print(X.shape)

np.random.shuffle(X) # so that validation data is not all from one category

################################################################################
# DEFINE DNN
################################################################################



input_img = Input(shape=desired_im_sz)
#print "shape of input", K.int_shape(input_img)

ks=5
x = Conv1D(64, kernel_size=ks, strides=(1), padding="same", activation="relu", data_format="channels_last")(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)
x = Conv1D(64, kernel_size=ks, strides=(1), padding='same', activation='relu', data_format="channels_last")(x)
x = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)
x = Conv1D(64, kernel_size=ks, strides=(1), padding="same", activation="relu", data_format="channels_last")(x)
x = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)
x = Conv1D(64, kernel_size=ks, strides=(1), padding="same", activation="relu", data_format="channels_last")(x)
encoded = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)

x = Conv1D(64, kernel_size=ks, strides=( 1), padding='same', activation='relu', data_format="channels_last")(encoded)
x = UpSampling1D(size=(4))(x)
# x = UpSampling2D(size=(4, 4), data_format="channels_first")(x) # for bottleneck of 25x25
x = Conv1D(64, kernel_size=ks, strides=( 1), padding='same', activation='relu', data_format="channels_last")(x)
x = UpSampling1D(size=(4))(x)
x = Conv1D(64, kernel_size=ks, strides=( 1), padding='same', activation='relu', data_format="channels_last")(x)
x = UpSampling1D(size=( 4))(x)
x = Conv1D(64, kernel_size=ks, strides=( 1), padding="same", activation="relu", data_format="channels_last")(x)
x = UpSampling1D(size=( 4))(x)

decoded = Conv1D(1, kernel_size=ks, strides=(1), padding="same", activation="sigmoid", data_format="channels_last")(x)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='mean_absolute_error', optimizer='adam')

print(autoencoder.summary())



######################################
# Train model
######################################
TrainProportion=0.95
ntrain= int(TrainProportion*X.shape[0])

X_val= X[ntrain:,:,:]
X= X[:ntrain,:,:]

weights_file = os.path.join( 'autoencoder_texture_weights_LMT_64_compression.hdf5')  # where weights will be saved
json_file = os.path.join( 'autoencoder_texture_model_LMT_64_compression.json')

csv_logger = CSVLogger(os.path.join('training_log_LMT_64_compression.csv'), append=True, separator=';') # added by KS
callbacks = [csv_logger]
callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size,
               shuffle=True, validation_data=(X_val, X_val), callbacks=callbacks, verbose=1)


#### save model keras
autoencoder.save('autoencoder_model_64_compression.h5')
encoder.save('encoder_model_64_compression.h5')
#print("Training finished, took {} hours.".format((time.time() - tic)/3600))


######################## 
# RECONTRUCT SIGNALS and save them as matlab arrays
# create output folders
if not os.path.exists('./recontructed_64_compression/'): os.mkdir('./recontructed_64_compression/')
if not os.path.exists('./Code_64_compression/'): os.mkdir('./Code_64_compression/')

import matplotlib.pyplot as plt
plt.figure()
for i in range(5):
    original = X[i]
    predicted=autoencoder.predict(np.expand_dims(original,0))
    plt.subplot(2,3, 1 + i)
    plt.plot(np.squeeze(original),'b')
    plt.plot(np.squeeze(predicted),'r')
    

Corrs_val=np.zeros(X_val.shape[0])
for i in range(X_val.shape[0]):
    tmp=X_val[i]
    tmp = np.expand_dims(tmp,0)
    tmphat=autoencoder.predict(tmp)
    #tmphat=tmphat[0,:,0]
    Corrs_val[i]=  np.mean((tmp-np.mean(tmp))*(tmphat-np.mean(tmphat)))/(np.std(tmphat)*np.std(tmp))
print(np.median(Corrs_val))

Corrs_train=np.zeros(X.shape[0])
for i in range(X.shape[0]):
    tmp=X[i]
    tmp = np.expand_dims(tmp,0)
    tmphat=autoencoder.predict(tmp)
    #tmphat=tmphat[0,:,0]
    Corrs_train[i]=  np.mean((tmp-np.mean(tmp))*(tmphat-np.mean(tmphat)))/(np.std(tmphat)*np.std(tmp))
print(np.median(Corrs_train))

# Save text file with ordered file names, as they are ordered when reconstructed - for later analyses
with open('imageListLMT_64_compression.txt', 'w') as f:
    for item in all_frames:
        f.write("%s\n" % item)





####### SAVE RECONTRASUCTED AND CODE
NAMES=list()  
X = np.zeros(( int(TotalSamples/2),) + desired_im_sz); # allocate space
counter=0;
for s in range(np.shape(all_frames)[0]):
    filename=all_frames[s]
    im=np.loadtxt(os.path.join(img_dir, filename), dtype='float', comments='#', delimiter=',')
                  
    ## LOW PASS- following weber
    im = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, im)
              
    #im_min=np.min(im[Begin:])
    #im_max=np.max(im[Begin:]-im_min)
    #im=(im-im_min)/im_max
    for ss in range(Nsubsamples):
        b=Begin+ss*desired_im_sz[0]
        end = b+desired_im_sz[0]
        SubSample= im[b:end]
        ###### notmalize (0 1)
        SubSample = SubSample-min(SubSample)
        SubSample = SubSample / max(SubSample)
        X[counter]=np.expand_dims(SubSample,1)
        NAMES.append([filename[0:(len(filename)-4)]+'_'+str(ss) + '.txt'])
        counter = counter+1
 
    


#code=encoder.predict(X)
for i in range(X.shape[0]):
    tmp=np.expand_dims(X[i,],0)
    code = encoder.predict(tmp)
    scipy.io.savemat("{}{}{}".format('./Code_64_compression/', NAMES[i], '.mat'), mdict={'arr': code})

for i in range(X.shape[0]):
    tmp=np.expand_dims(X[i,],0)
    tmp = autoencoder.predict(tmp)
    scipy.io.savemat("{}{}{}".format('./recontructed_64_compression/', NAMES[i], '.mat'), mdict={'arr': tmp})










