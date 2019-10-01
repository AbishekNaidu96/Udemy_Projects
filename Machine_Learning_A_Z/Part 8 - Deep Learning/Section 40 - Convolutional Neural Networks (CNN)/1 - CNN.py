# CONVOLUTIONAL NEURAL NETWORKS

#------------------------------------------------------------------------------------#
#----------------------------- PART I: BUILDING THE CNN -----------------------------#
#------------------------------------------------------------------------------------#
# Importing the libraries
from keras.models import Sequential                    
# Used to initialize the neural network. There are two ways of initializing
    # 1. As a SEQUENCE OF LAYERS - A CNN is asequence of layers
    # 2. AS A GRAPH
from keras.layers import Convolution2D
# First step of making the CNN, adding the covolutional layers
# Images are in 2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Package to add the FULLY CONNECTED LAYERS in our CNN

# Initialising the CNN
classifier = Sequential()


#--------------- STEP I: CONVOLUTION ---------------#
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# NB_FILTER -> the number of filters to be used (that many number of feature maps would
        # be created). It is best practise to start with 32 then add on to get 64 / 128
# NB_ROW / NB_COL -> number of rows and columsn in the feature detector (FILTER MATRIX)
# INPUT_SHAPE = (NUMBER OF CHANNELS, 3 for color - RBG, 1 for B/W
        # 256 / 256 (64, 64 HERE) are the dimensions of the 2D image)
        # THE ORDER GIVEN ABOVE IS THE ORDER FOR THEANO
        # HOWEVER, WE ARE USING TENSORFLOW, SO THE ORDER WOULD CHANGE 
        
#--------------- STEP II: POOLING ---------------#
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#------------- STEP III: FLATTENING -------------#
classifier.add(Flatten())

#----------- STEP IV: FULL CONNECTION -----------#
# INPUT LAYER
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# OUTPUT_DIM = 128; because we choose a value that is not too small to make the 
            #clssifier a good model and also not too big to make it highly compute 
            #intense
    # Remember  that there are 32 filters, creating 32 feature maps that would be 
    # flattened and used as input dimensions for the FULL CONNECTION
# OUTPUT LAYER
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#--------------- COMPILING THE CNN ---------------#
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])



#------------------------------------------------------------------------------------#
#---------------------- PART II: FITTING THE CNN TO THE IMAGES ----------------------#
#------------------------------------------------------------------------------------#
from keras.preprocessing.image import ImageDataGenerator

# We generate a population of images by changing the images we have, by rotating
        # SHEARING, zooming, flipping the image
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size = (150, 150),
        batch_size = 32,
        class_mode = 'binary')

model.fit_generator(
        train_generator,
        steps_per_epoch = 2000,
        epochs = 50,
        validation_data = validation_generator,
        validation_steps = 800)



























