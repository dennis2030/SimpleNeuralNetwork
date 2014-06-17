Usage: Use 'matlab < train.m' for training, 'matlab < test.m' for testing.

All the parameters are in 'train.json', which is a configuration file encoded with json string.
You can modify all the parameters to change how the NN work.
For example, 'preTrain':1 means do pre-training first. Set the value to 0 would skip the pre-training phase.


Implementation:

All the implementation are follow the tutorial in (' http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial ')
It would use sparse autoencoder to do pre-training, and training with simple forward and backward propagation.
Also, momentum and learning rate decay are apply to it.

We implement 2 activation function, you can specify it in train.json.
'activateType':1 means sigmoid, 2 means tanh.

For more information, you could contact me via dennis2030@gmail.com.
