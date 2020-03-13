import tensorflow as tf
# print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

# normalise the data because NN take number from 0-1 better than 0-255
training_images  = training_images / 255.0
test_images = test_images / 255.0

# training_images  = training_images 
# test_images = test_images 

# adding callback functino to the model training process to stop training the model when the ide
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# use three layers to build the model 
# flatten transfer the 28x28 matrix to a 56x1 array
# nn.relu (If X>0 return X, else return 0") only pass numnber larger than 0 to next layer
## we use 128 neuron to build the model
## the more we use the final answer is more accurate but longer trainning time
# nn.softmax (takes a set of values, and effectively picks the biggest one) in this case 
## only return the label with the maximum possibility
## also we only use 10 neuron to perform classification which should match the number of training tags
### the number of middle layer: 1, 10, 64, 128, 512, 1024 ...) how to choose ?

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# ERROR: module 'tensorflow_core._api.v2.train' has no attribute 'AdamOptimizer'
# model.compile(optimizer = tf.train.AdamOptimizer()
# SOLUTION: directly use optimizer to 'adam'

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# increase the epochs may increase the accuracy of the model on training data
# but the loss value will increase becasue of overfitting
# model.fit(training_images, training_labels, epochs=5)

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

model.evaluate(test_images,test_labels)

classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])