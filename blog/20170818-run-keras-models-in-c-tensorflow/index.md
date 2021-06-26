# Run your Keras models in C++ Tensorflow
## Aug 18, 2017
:warning: :warning: This tutorial was new in August of 2017 - it is now ancient, and many of these libraries and processes have changed - please keep that in mind as you read / follow :warning: :warning:

So you’ve built an awesome machine learning model in Keras and now you want to run it natively thru Tensorflow. This tutorial will show you how. All of the code in this tutorial can be cloned / downloaded from https://github.com/bitbionic/keras-to-tensorflow.git . You may want to clone it to follow along.

Keras is a wonderful high level framework for building machine learning models. It is able to utilize multiple backends such as Tensorflow or Theano to do so. When a Keras model is saved via the .save method, the canonical save method serializes to an HDF5 format. Tensorflow works with Protocol Buffers, and therefore loads and saves .pb files. This tutorial demonstrates how to:

 * build a SIMPLE Convolutional Neural Network in Keras for image classification
 * save the Keras model as an HDF5 model
 * verify the Keras model
 * convert the HDF5 model to a Protocol Buffer
 * build a Tensorflow C++ shared library
 * utilize the .pb in a pure Tensorflow app
 * We will utilize Tensorflow’s own example code for this
 * I am conducting this tutorial on Linux Mint 18.1, using GPU accelerated Tensorflow version 1.1.0 and Keras version 2.0.4. I have run this on Tensorflow v.1.3.0 as well.

```
Python 3.6.1 |Anaconda custom (64-bit)| (default, May 11 2017, 13:09:58) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow, keras
Using TensorFlow backend.
>>> tensorflow.__version__
'1.1.0'
>>> keras.__version__
'2.0.4'
```

A NOTE ABOUT WINDOWS: Everything here SHOULD work on Windows as well until we reach C++. Building Tensorflow on Windows is a bit different (and to this point a bit more challenging) and I haven’t fully vetted the C++ portion of this tutorial on Windows yet.  I will update this post upon vetting Windows.

## Assumptions
 * You are familiar with Python (and C++ if you’re interested in the C++ portion of this tutorial)
 * You are familiar with Keras and Tensorflow and already have your dev environment setup
 * Example code is utilizing Python 3.5, if you are using 2.7 you may have to make modifications

## Get a dataset

I’m assuming that if you’re interested in this topic  you probably already have some image classification data. You may use that or follow along with this tutorial where we use the flowers data from the Tensorflow examples. It’s about 218 MB and you can download it from http://download.tensorflow.org/example_images/flower_photos.tgz

After extracting the data you should see a folder structure similar to the image shown here. There are 5 categories and the data is pre-sorted into test and train.

## Train your model
I will use a VERY simple CNN for this example, however the techniques to port the models work equally well with the built-in Keras models such as Inception and ResNet. I have no illusions that this model will win any awards, but it will serve our purpose.

There are a few things to note from the code listed below:

 * Label your input and output layer(s) – this will make it easier to debug when the model is converted.
 * I’m relying on the Model Checkpoint to save my .h5 files – you could also just call classifier.save after the training is complete.
 * Make note of the shape parameter you utilize, we will need that when we run the model later.


### k2tf_trainer.py
```python
'''
This script builds and trains a simple Convolutional Nerual Network (CNN)
against a supplied data set. It is used in a tutorial demonstrating
how to build Keras models and run them in native C++ Tensorflow applications.

MIT License

Copyright (c) 2017 bitbionic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from datetime import datetime
import os
import argparse

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ZeroPadding2D

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def buildClassifier( img_shape=128, num_categories=5 ):
    '''
    Builds a very simple CNN outputing num_categories.
    
    Args:
             img_shape (int): The shape of the image to feed the CNN - defaults to 128
        num_categories (int): The number of categories to feed the CNN
 
    Returns:
        keras.models.Model: a simple CNN
    
    '''
    classifier = Sequential()

    # Add our first convolutional layer
    classifier.add( Conv2D( filters=32,
                            kernel_size=(2,2),
                            padding='same',
                            data_format='channels_last',
                            input_shape=(img_shape,img_shape,3),
                            activation = 'relu',
                            name = 'firstConv2D'
                            ) )

    # Pooling
    classifier.add( MaxPooling2D(pool_size=(2,2), name='firstMaxPool') )

    # Add second convolutional layer.
    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=16,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'secondConv2D'
                            ) 
                  )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='secondMaxPool') )
    
    # Add second convolutional layer.
    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=8,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'thirdc2'
                            ) 
                  )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='thirdpool') )


    # Flattening
    classifier.add( Flatten(name='flat') )

    # Add Fully connected ANN
    classifier.add( Dense( units=256, activation='relu', name='fc256') )
    classifier.add( Dense( units=num_categories, activation = 'softmax', name='finalfc'))

    # Compile the CNN
    #classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


def trainModel( classifier, trainloc, testloc, img_shape, output_dir='./', batch_size=32, num_epochs=30 ):
    '''
    Trains the supplied model agaist train and test locations specified
    in the args. During the training, each epoch will be evaluated for 
    val_loss and the model will be saved if val_loss is lower than
    previous.
    
    Args:
        classifier (keras.models.Model): the model to be trained.
                         trainloc (str): the location of the training data
                          testloc (str): the location of the test data
                        img_shape (int): the shape of the image to feed the CNN
                       output_dir (str): the directory where output files are saved
                       batch_size (int): the number of samples per gradient update
                       num_epochs (int): the number of epochs to train a model
    
    Returns:
        keras.models.Model: returns the trained CNN
    '''
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2, 
                                       rotation_range=25,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set = train_datagen.flow_from_directory(trainloc,
                                                     target_size = (img_shape, img_shape),
                                                     batch_size = batch_size,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(testloc,
                                                target_size = (img_shape, img_shape),
                                                batch_size = batch_size,
                                                class_mode = 'categorical')

    # Saves the model weights after each epoch if the validation loss decreased
    now = datetime.now()
    nowstr = now.strftime('k2tf-%Y%m%d%H%M%S')

    now = os.path.join( output_dir, nowstr)

    # Make the directory
    os.makedirs( now, exist_ok=True )

    # Create our callbacks
    savepath = os.path.join( now, 'e-{epoch:03d}-vl-{val_loss:.3f}-va-{val_acc:.3f}.h5' )
    checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    fout = open( os.path.join(now, 'indices.txt'), 'wt' )
    fout.write( str(training_set.class_indices) + '\n' )

    # train the model on the new data for a few epochs
    classifier.fit_generator(training_set,
                             steps_per_epoch = len(training_set.filenames)//batch_size,
                             epochs = num_epochs,
                             validation_data = test_set,
                             validation_steps = len(test_set.filenames)//batch_size,
                             workers=32, 
                             max_q_size=32,
                             callbacks=[checkpointer]
                             )
    
    return classifier


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Required
    parser.add_argument('--test', dest='test', required=True, help='(REQUIRED) location of the test directory')
    parser.add_argument('--train', dest='train', required=True, help='(REQUIRED) location of the test directory')
    parser.add_argument('--cats', '-c', dest='categories', type=int, required=True, help='(REQUIRED) number of categories for the model to learn')
    # Optional
    parser.add_argument('--output', '-o', dest='output', default='./', required=False, help='location of the output directory (default:./)')
    parser.add_argument('--batch', '-b', dest='batch', default=32, type=int, required=False, help='batch size (default:32)')
    parser.add_argument('--epochs', '-e', dest='epochs', default=30, type=int, required=False, help='number of epochs to run (default:30)')
    parser.add_argument('--shape','-s', dest='shape', default=128, type=int, required=False, help='The shape of the image, single dimension will be applied to height and width (default:128)')
    
    args = parser.parse_args()
    
    classifier = buildClassifier( args.shape, args.categories)
    trainModel( classifier, args.train, args.test, args.shape, args.output, batch_size=args.batch, num_epochs=args.epochs )
```

I down-sampled the imagery significantly and ran the model more than I needed to, but here was the command I ran (NOTE: I ran this on some old hardware using GPU acceleration on a NVIDIA GTX-660 – you can probably increase the batch size significantly assuming you have better hardware):

```
python k2tf_trainer.py -h
Using TensorFlow backend.
usage: k2tf_trainer.py [-h] --test TEST --train TRAIN --cats CATEGORIES
                       [--output OUTPUT] [--batch BATCH] [--epochs EPOCHS]
                       [--shape SHAPE]

optional arguments:
  -h, --help            show this help message and exit
  --test TEST           (REQUIRED) location of the test directory
  --train TRAIN         (REQUIRED) location of the test directory
  --cats CATEGORIES, -c CATEGORIES
                        (REQUIRED) number of categories for the model to learn
  --output OUTPUT, -o OUTPUT
                        location of the output directory (default:./)
  --batch BATCH, -b BATCH
                        batch size (default:32)
  --epochs EPOCHS, -e EPOCHS
                        number of epochs to run (default:30)
  --shape SHAPE, -s SHAPE
                        The shape of the image, single dimension will be
                        applied to height and width (default:128)

python k2tf_trainer.py --test=../../data/flowers/raw-data/validation --train=../../data/flowers/raw-data/train --cats=5 --shape=80 --batch=120 --epochs=400 --output=./temp
```

A few runs of this yielded val_acc in the 83-86% range, and while it’s no Inception, it’s good enough for this exercise.

## Test your model
So now let’s just do a quick gut-check on our model – here’s a small script to load your model, image, shape and indices (especially if you didn’t use the flowers set):

### k2tf_trainer.py
```python

'''
This script evaluates a simple Convolutional Nerual Network (CNN)
against a supplied data set. It is used in a tutorial demonstrating
how to build Keras models and run them in native C++ Tensorflow applications.

MIT License

Copyright (c) 2017 bitbionic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import argparse

import numpy as np

from keras.preprocessing import image
from keras.models import load_model

def invertKVPairs( someDictionary ):
    '''
    Inverts the key/value pairs of the supplied dictionary.
    
    Args:
        someDictionary (dict): The dictionary for which you would like the inversion
    Returns:
        Dictionary - the inverse key-value pairing of someDictionary
    '''
    ret = {}
    for k, v in someDictionary.items():
        ret[v] = k

    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', dest='model', required=True, help='The HDF5 Keras model you wish to run')
    parser.add_argument('--image','-i', dest='image', required=True, help='The image you wish to test')
    parser.add_argument('--shape','-s', type=int, dest='shape', required=True, help='The shape to resize the image for the model')
    parser.add_argument('--labels','-l', dest='labels', required=False, help='The indices.txt file containing the class_indices of the Keras training set')
    args = parser.parse_args()

    model = load_model(args.model)
    
    # These indices are saved on the output of our trainer
    class_indices = { 0:'daisy',  1:'dandelion',  2:'roses', 3:'sunflowers', 4:'tulips' }
    if args.labels:
        with open( args.labels, 'rt' ) as infile:
            label_str = infile.read()
            str2dict = eval(label_str)
            class_indices = invertKVPairs( str2dict )

    test_image = image.load_img(args.image, target_size=(args.shape,args.shape))
    test_image = np.expand_dims( test_image, axis=0 )
    test_image = test_image/255.0
    result = model.predict(test_image)

    for idx,val in enumerate( result[0] ):
        print( '%s : %4.2f percent' % (class_indices[idx], val*100. ) )
```

Here’s a few examples of my runs for reference:

```
python k2tf_eval.py -h
Using TensorFlow backend.
usage: k2tf_eval.py [-h] --model MODEL --image IMAGE --shape SHAPE
                    [--labels LABELS]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        The HDF5 Keras model you wish to run
  --image IMAGE, -i IMAGE
                        The image you wish to test
  --shape SHAPE, -s SHAPE
                        The shape to resize the image for the model
  --labels LABELS, -l LABELS
                        The indices.txt file containing the class_indices of
                        the Keras training set
######################
# LETS TRY A DANDELION
######################
python k2tf_eval.py -m '/home/bitwise/Development/keras-to-tensorflow/temp/k2tf-20170816140235/e-075-vl-0.481-va-0.837.h5' -i '/home/bitwise/data/flowers/raw-data/validation/dandelion/13920113_f03e867ea7_m.jpg' -s 80
Using TensorFlow backend.

daisy : 0.02 percent
dandelion : 99.95 percent
roses : 0.02 percent
sunflowers : 0.00 percent
tulips : 0.01 percent

######################
# LETS TRY A ROSE
######################
python k2tf_eval.py -m '/home/bitwise/Development/keras-to-tensorflow/temp/k2tf-20170816140235/e-075-vl-0.481-va-0.837.h5' -i '/home/bitwise/data/flowers/raw-data/validation/roses/160954292_6c2b4fda65_n.jpg' -s 80
Using TensorFlow backend.

daisy : 1.85 percent
dandelion : 0.14 percent
roses : 70.67 percent
sunflowers : 0.01 percent
tulips : 27.33 percent


######################
# LETS TRY A TULIP
######################
python k2tf_eval.py -m '/home/bitwise/Development/keras-to-tensorflow/temp/k2tf-20170816140235/e-075-vl-0.481-va-0.837.h5' -i '/home/bitwise/data/flowers/raw-data/validation/tulips/450607536_4fd9f5d17c_m.jpg' -s 80
Using TensorFlow backend.

daisy : 0.00 percent
dandelion : 0.00 percent
roses : 1.65 percent
sunflowers : 0.52 percent
tulips : 97.83 percent
```

Alright – not too bad, now for the fun part.

## Convert from HDF5 to .pb
### Attribution: This script was adapted from https://github.com/amir-abdi/keras_to_tensorflow

I adapted the notebook from the link above to a script we can run from the command line. The code is almost identical except for the argument parsing. This code does the following:

 * Loads your .h5 file
 * Replaces your output tensor(s) with a named Identity Tensor – this can be helpful if you are using a model you didn’t build and don’t know all of the output names (of course you could go digging, but this avoids that).
 * Saves an ASCII representation of the graph definition. I use this to verify my input and output names for Tensorflow. This can be useful in debugging.
 * Replaces all variables within the graph to constants.
 * Writes the resulting graph to the output name you specify in the script.
 * With that said here’s the code:
 ** An error has occurred. Please try again later.

When you run the code, it’s important that you make the prefix name unique to the graph. If you didn’t build the graph, using something like “output” or some other generic name has the potential of colliding with a node of the same name within the graph. I recommend making the prefix name uniquely identifiable, and for that reason this script defaults the prefix to “k2tfout” though you can override that with whatever you prefer.

And now let’s run this little guy on our trained model.

```
Using TensorFlow backend.
usage: k2tf_convert.py [-h] --model MODEL --numout NUM_OUT [--outdir OUTDIR]
                       [--prefix PREFIX] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        REQUIRED: The HDF5 Keras model you wish to convert to
                        .pb
  --numout NUM_OUT, -n NUM_OUT
                        REQUIRED: The number of outputs in the model.
  --outdir OUTDIR, -o OUTDIR
                        The directory to place the output files -
                        default("./")
  --prefix PREFIX, -p PREFIX
                        The prefix for the output aliasing -
                        default("k2tfout")
  --name NAME           The name of the resulting output graph -
                        default("output_graph.pb")



python k2tf_convert.py -m '/home/bitwise/Development/keras-to-tensorflow/temp/k2tf-20170816140235/e-075-vl-0.481-va-0.837.h5' -n 1
Using TensorFlow backend

Output nodes names are:  ['k2tfout_0']
Saved the graph definition in ascii format at:  ./graph_def_for_reference.pb.ascii
Converted 10 variables to const ops.
Saved the constant graph (ready for inference) at:  ./output_graph.pb
```

As you can see, two files were written out. An ASCII and .pb file. Let’s look at the graph structure, notice the input node name “firstConv2D_input” and the output name “k2tfout_0”, we will use those in the next section:

### graph_def_for_reference.pb.ascii
```javascript
node {
  name: "firstConv2D_input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
//.
//.  Many more nodes here
//.
//.
node {
  name: "k2tfout_0"
  op: "Identity"
  input: "strided_slice"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 21
}
```

## Use Tensorflow’s label_image examples:
The remainder of this tutorial will heavily leverage Tensorflow’s image recognition examples. Specifically this file for python and this file for C++.

I copied both of those files into the git repo for this tutorial. Now let’s test them out.

## Running your Tensorflow model with Python
Running the Python script is fairly straight forward. Remember, we need to supply the following arguments:

 * the output_graph.pb we generated above
 * the labels file – this is supplied with the dataset but you could generate a similar labels.txt from the indices.txt file we produced in our Keras model training
 * input width and height. Remember I trained with 80×80 so I must adjust for that here
 * The input layer name – I find this in the generated ASCII file from the conversion we did above. In this case it is “firstConv2D_input” – Remember our k2tf_trainer.py named the first layer “firstConv2D”.
 * The output layer name – We created this with prefix and can verify it in our ASCII file. We went with the script default which was “k2tfout_0”
 * Finally, the image we want to process.
 * 
Let’s try it:

### Build instructions
```
python label_image.py -h
usage: label_image.py [-h] [--image IMAGE] [--graph GRAPH] [--labels LABELS]
                      [--input_height INPUT_HEIGHT]
                      [--input_width INPUT_WIDTH] [--input_mean INPUT_MEAN]
                      [--input_std INPUT_STD] [--input_layer INPUT_LAYER]
                      [--output_layer OUTPUT_LAYER]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         image to be processed
  --graph GRAPH         graph/model to be executed
  --labels LABELS       name of file containing labels
  --input_height INPUT_HEIGHT
                        input height
  --input_width INPUT_WIDTH
                        input width
  --input_mean INPUT_MEAN
                        input mean
  --input_std INPUT_STD
                        input std
  --input_layer INPUT_LAYER
                        name of input layer
  --output_layer OUTPUT_LAYER
                        name of output layer

######################
# LETS TRY A DANDELION
######################
python label_image.py --graph=./output_graph.pb --labels=../../data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=../../data/flowers/raw-data/validation/dandelion/13920113_f03e867ea7_m.jpg 
2017-08-18 13:05:04.977442: I ./main.cpp:250] dandelion (1): 0.999625
2017-08-18 13:05:04.977517: I ./main.cpp:250] daisy (0): 0.00023931
2017-08-18 13:05:04.977530: I ./main.cpp:250] roses (2): 8.84324e-05
2017-08-18 13:05:04.977541: I ./main.cpp:250] tulips (4): 4.70001e-05
2017-08-18 13:05:04.977553: I ./main.cpp:250] sunflowers (3): 7.26905e-08

######################
# LETS TRY A ROSE
######################
python label_image.py --graph=./output_graph.pb --labels=../../data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=../../data/flowers/raw-data/validation/roses/160954292_6c2b4fda65_n.jpg
2017-08-18 13:06:27.205764: I ./main.cpp:250] roses (2): 0.660571
2017-08-18 13:06:27.205834: I ./main.cpp:250] tulips (4): 0.31912
2017-08-18 13:06:27.205845: I ./main.cpp:250] daisy (0): 0.0188609
2017-08-18 13:06:27.205853: I ./main.cpp:250] dandelion (1): 0.00124337
2017-08-18 13:06:27.205862: I ./main.cpp:250] sunflowers (3): 0.000204971


######################
# LETS TRY A TULIP
######################
python label_image.py --graph=./output_graph.pb --labels=../../data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=../../data/flowers/raw-data/validation/tulips/450607536_4fd9f5d17c_m.jpg
2017-08-18 13:08:08.395576: I ./main.cpp:250] tulips (4): 0.96644
2017-08-18 13:08:08.395657: I ./main.cpp:250] sunflowers (3): 0.0188329
2017-08-18 13:08:08.395669: I ./main.cpp:250] roses (2): 0.0147128
2017-08-18 13:08:08.395681: I ./main.cpp:250] daisy (0): 1.08454e-05
2017-08-18 13:08:08.395691: I ./main.cpp:250] dandelion (1): 3.82244e-06
```

So now Tensorflow is running our model in Python – but how do we get to C++?

## Running your Tensorflow model with C++
If you are still reading then I’m assuming you need to figure out how to run Tensorflow in a production environment on C++. This is where I landed, and I had to bounce between fragments of tutorials to get things to work. Hopefully the information here will give you a consolidated view of how to accomplish this.

For my project, I wanted to have a Tensorflow shared library that I could link and deploy. That’s what we’ll build in this project and build the label_image example with it.

To run our models in C++ we first need to obtain the Tensorflow source tree. The instructions are here, but we’ll walk thru them below.

### Clone the repo
```
(tensorflow) $ ~/Development/keras-to-tensorflow $ git clone https://github.com/tensorflow/tensorflow
Cloning into 'tensorflow'...
remote: Counting objects: 223667, done.
remote: Compressing objects: 100% (15/15), done.
remote: Total 223667 (delta 2), reused 8 (delta 0), pack-reused 223652
Receiving objects: 100% (223667/223667), 116.84 MiB | 3.17 MiB/s, done.
Resolving deltas: 100% (173085/173085), done.
Checking connectivity... done.
(tensorflow) $ ~/Development/keras-to-tensorflow $ cd tensorflow/
(tensorflow) $ ~/Development/keras-to-tensorflow/tensorflow $ git checkout r1.1
Branch r1.1 set up to track remote branch r1.1 from origin.
Switched to a new branch 'r1.1'
(tensorflow) $ ~/Development/keras-to-tensorflow/tensorflow $ ls
ACKNOWLEDGMENTS  BUILD              LICENSE       tensorflow   WORKSPACE
ADOPTERS.md      configure          models.BUILD  third_party
AUTHORS          CONTRIBUTING.md    README.md     tools
bower.BUILD      ISSUE_TEMPLATE.md  RELEASE.md    util
(tensorflow) $ ~/Development/keras-to-tensorflow/tensorflow $
```

Now that we have the source code, we need the tools to build it. On Linux or Mac Tensorflow uses Bazel. Windows uses CMake (I tried using Bazel on Windows but was not able to get it to work).  Again installation instructions for Linux are here, and Mac here, but we’ll walk thru the Linux instructions below. Of course there may be some other dependencies but I’m assuming if you’re taking on building Tensorflow, this isn’t your first rodeo.

Install Python dependencies (I’m using 3.x, for 2.x omit the ‘3’)
```
sudo apt-get install build-essential python3-numpy python3-dev python3-pip python3-wheel
```
Install JDK 8, you can use either Oracle or OpenJDK. I’m using openjdk-8 on my system (in fact I think I already had it installed). If you don’t, simply type:

Install JDK 8
```
sudo apt-get install openjdk-8-jdk
```

NOTE: I have not tested building with CUDA – this is just the documentation that I’ve read. For deployment I didn’t want to build with CUDA, however if you do then you of course need the CUDA SDK and the CUDNN code from NVIDIA. You’ll also need to grab libcupti-dev.

Next, lets install Bazel:

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel
```

At this point you should be able to run bazel help and get feedback:

```
(tensorflow) $ ~/Development/keras-to-tensorflow/tensorflow $ bazel help
.......................................
                                                           [bazel release 0.5.3]
Usage: bazel <command> <options> ...

Available commands:
  analyze-profile     Analyzes build profile data.
  build               Builds the specified targets.
  canonicalize-flags  Canonicalizes a list of bazel options.
  clean               Removes output files and optionally stops the server.
  coverage            Generates code coverage report for specified test targets.
  dump                Dumps the internal state of the bazel server process.
  fetch               Fetches external repositories that are prerequisites to the targets.
  help                Prints help for commands, or the index.
  info                Displays runtime info about the bazel server.
  license             Prints the license of this software.
  mobile-install      Installs targets to mobile devices.
  query               Executes a dependency graph query.
  run                 Runs the specified target.
  shutdown            Stops the bazel server.
  test                Builds and runs the specified test targets.
  version             Prints version information for bazel.

Getting more help:
  bazel help <command>
                   Prints help and options for <command>.
  bazel help startup_options
                   Options for the JVM hosting bazel.
  bazel help target-syntax
                   Explains the syntax for specifying targets.
  bazel help info-keys
                   Displays a list of keys used by the info command.
```

Now that we have everything installed, we can configure and build. Make sure you’re in the top-level tensorflow directory. I went with all the default configuration options. When you do this, the configuration tool will download a bunch of dependencies – this make take a minute or two.

### Configure Bazel
```
(tensorflow) $ ~/Development/keras-to-tensorflow/tensorflow $ ./configure 
Please specify the location of python. [Default is /home/bitwise/anaconda3/envs/tensorflow/bin/python]: 
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
Do you wish to use jemalloc as the malloc implementation? [Y/n] 
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] 
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] 
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] 
No XLA support will be enabled for TensorFlow
Found possible Python library paths:
  /home/bitwise/anaconda3/envs/tensorflow/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/bitwise/anaconda3/envs/tensorflow/lib/python3.6/site-packages]

Using python library path: /home/bitwise/anaconda3/envs/tensorflow/lib/python3.6/site-packages
Do you wish to build TensorFlow with OpenCL support? [y/N] 
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] 
No CUDA support will be enabled for TensorFlow
Configuration finished
INFO: Starting clean (this may take a while). Consider using --expunge_async if the clean takes more than several minutes.
............
WARNING: ~/.cache/bazel/_bazel_bitwise/7f7ca38846ebb2cb18e80e7c35ca353a/external/bazel_tools/tools/build_defs/pkg/pkg.bzl:196:9: pkg_tar: renaming non-dict `files` attribute to `srcs`
WARNING: ~/.cache/bazel/_bazel_bitwise/7f7ca38846ebb2cb18e80e7c35ca353a/external/bazel_tools/tools/build_defs/pkg/pkg.bzl:196:9: pkg_tar: renaming non-dict `files` attribute to `srcs`
WARNING: ~/.cache/bazel/_bazel_bitwise/7f7ca38846ebb2cb18e80e7c35ca353a/external/bazel_tools/tools/build_defs/pkg/pkg.bzl:196:9: pkg_tar: renaming non-dict `files` attribute to `srcs`
WARNING: ~/.cache/bazel/_bazel_bitwise/7f7ca38846ebb2cb18e80e7c35ca353a/external/bazel_tools/tools/build_defs/pkg/pkg.bzl:196:9: pkg_tar: renaming non-dict `files` attribute to `srcs`
Building: no action running
```

Alright, we’re configured, now it’s time to build.  DISCLAIMER: I AM NOT a Bazel guru. I found these settings via Google-Fu and digging around in the configuration files. I could not find a way to get Bazel to dump the available targets. Most tutorials I saw are about building the pip target – however, I wanted to build a .so.  I started looking thru the BUILD files to find targets and found these in tensorflow/tensorflow/BUILD

### Tensorflow Targets
```python
cc_binary(
    name = "libtensorflow.so",
    linkshared = 1,
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "libtensorflow_cc.so",
    linkshared = 1,
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/core:tensorflow",
    ],
)
```

So with that in mind here’s the command for doing this (you may want to alter jobs based on your number of cores and RAM – also you can remove avx, mfpmath and msse4.2 optimizations if you wish):

## Build our Tensorflow shared library
```
bazel build --jobs=6 --verbose_failures -c opt --copt=-mavx --copt=-mfpmath=both --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
```
Go get some coffee, breakfast, lunch or watch a show. This will grind for a while, but you should end up with bazel-bin/tensorflow/libtensorflow_cc.so.

## Let’s run it!
In the tutorial git repo, I’m including a qmake .pro file that links the .so and all of the required header locations. I’m including it for reference – you DO NOT need qmake to build this. In fact, I’m including the g++ commands to build. You may have to adjust for your environment. Assuming you’re building main.cpp from the root of the repo, and the tensorflow build we just created was cloned and built in the same directory, all paths should be relative and work out of the box.

### Tutorial.pro qmake build file
```
TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += $$PWD/tensorflow
INCLUDEPATH += $$PWD/tensorflow/bazel-tensorflow/external/eigen_archive
INCLUDEPATH += $$PWD/tensorflow/bazel-tensorflow/external/protobuf/src
INCLUDEPATH += $$PWD/tensorflow/bazel-genfiles

LIBS += -L$$PWD/tensorflow/bazel-bin/tensorflow -ltensorflow_cc

SOURCES += main.cpp
```

I included a build_tutorial_cpp.sh, here are the commands:

### Build instructions
```
g++ -c -pipe -g -std=gnu++11 -Wall -W -fPIC -I. -I./tensorflow -I./tensorflow/bazel-tensorflow/external/eigen_archive -I./tensorflow/bazel-tensorflow/external/protobuf/src -I./tensorflow/bazel-genfiles -o main.o ./main.cpp
g++  -o Tutorial main.o   -L./tensorflow/bazel-bin/tensorflow -ltensorflow_cc
cp ./tensorflow/bazel-bin/tensorflow/libtensorflow* .
```

Now you should have an executable in your directory and we can now test our application:

```
./Tutorial -h
2017-08-18 12:57:11.656298: E ./main.cpp:318] Unknown argument -h
usage: ./Tutorial
Flags:
	--image="tensorflow/examples/label_image/data/grace_hopper.jpg"	string	image to be processed
	--graph="tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"	string	graph to be executed
	--labels="tensorflow/examples/label_image/data/imagenet_slim_labels.txt"	string	name of file containing labels
	--input_width=299                	int32	resize image to this width in pixels
	--input_height=299               	int32	resize image to this height in pixels
	--input_layer="input"            	string	name of input layer
	--output_layer="InceptionV3/Predictions/Reshape_1"	string	name of output layer
	--self_test=false                	bool	run a self test
	--root_dir=""                    	string	interpret image and graph file names relative to this directory

######################
# LETS TRY A DANDELION
######################
./Tutorial --graph=./output_graph.pb --labels=../../data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=../../data/flowers/raw-data/validation/dandelion/13920113_f03e867ea7_m.jpg 
2017-08-18 13:05:04.977442: I ./main.cpp:250] dandelion (1): 0.999625
2017-08-18 13:05:04.977517: I ./main.cpp:250] daisy (0): 0.00023931
2017-08-18 13:05:04.977530: I ./main.cpp:250] roses (2): 8.84324e-05
2017-08-18 13:05:04.977541: I ./main.cpp:250] tulips (4): 4.70001e-05
2017-08-18 13:05:04.977553: I ./main.cpp:250] sunflowers (3): 7.26905e-08

######################
# LETS TRY A ROSE
######################
./Tutorial --graph=./output_graph.pb --labels=../../data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=../../data/flowers/raw-data/validation/roses/160954292_6c2b4fda65_n.jpg
2017-08-18 13:06:27.205764: I ./main.cpp:250] roses (2): 0.660571
2017-08-18 13:06:27.205834: I ./main.cpp:250] tulips (4): 0.31912
2017-08-18 13:06:27.205845: I ./main.cpp:250] daisy (0): 0.0188609
2017-08-18 13:06:27.205853: I ./main.cpp:250] dandelion (1): 0.00124337
2017-08-18 13:06:27.205862: I ./main.cpp:250] sunflowers (3): 0.000204971


######################
# LETS TRY A TULIP
######################
./Tutorial --graph=./output_graph.pb --labels=../../data/flowers/raw-data/labels.txt --input_width=80 --input_height=80 --input_layer=firstConv2D_input --output_layer=k2tfout_0 --image=../../data/flowers/raw-data/validation/tulips/450607536_4fd9f5d17c_m.jpg
2017-08-18 13:08:08.395576: I ./main.cpp:250] tulips (4): 0.96644
2017-08-18 13:08:08.395657: I ./main.cpp:250] sunflowers (3): 0.0188329
2017-08-18 13:08:08.395669: I ./main.cpp:250] roses (2): 0.0147128
2017-08-18 13:08:08.395681: I ./main.cpp:250] daisy (0): 1.08454e-05
2017-08-18 13:08:08.395691: I ./main.cpp:250] dandelion (1): 3.82244e-06
```

So there we have it. I did notice that the percentages aren’t exactly the same as when I ran the Keras models directly. I’m not sure if this is a difference in compiler settings or if Keras is overriding some calculations. But this gets you well on your way to running a Keras model in C++.

I hope this was useful to you.
