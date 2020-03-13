import tensorflow as tf
import matplotlib.pyplot as plt

# import data define the path
# no need to specific the path when load the data blow
path = f"/Users/stark/Desktop/TensorFlow/mnist.npz"

#impletfy the callback function to stop the training when optimal accuracy reached
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.90):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

mnist = tf.keras.datasets.mnist

df = mnist.load_data()
# df = mnist.load_data(path = path)


(x_train, y_train),(x_test, y_test) = df

# this is not a gray scale image why we are using 256 to nomalise the data?
x_train = x_train / 255.0
x_test = x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

history.epoch, history.history['accuracy'][-1]

predict_number = model.predict(x_test)

print(predict_number[0],y_test[0])

plt.imshow(x_test[0])
plt.show()