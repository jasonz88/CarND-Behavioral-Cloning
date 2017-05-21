import os
import csv
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, SpatialDropout2D, Flatten, Lambda, Cropping2D, Reshape
from keras.optimizers import Adam
from scipy.misc import imread, imresize
import numpy as np
import sklearn
from sklearn.utils import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'C:/Users/dyz/Downloads/data/data/IMG/', 'The directory of the image data.')
flags.DEFINE_string('csv_path', 'C:/Users/dyz/Downloads/data/data/driving_log.csv', 'The path to the csv of training data.')
flags.DEFINE_string('csv_path_augmented', 'C:/Users/dyz/Desktop/driving_log.csv', 'The path to the csv of training data.')
flags.DEFINE_integer('batch_size',128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 13, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.001, 'The learning rate for training.')


def generator(samples, batch_size=FLAGS.batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if batch_sample[0][0:3] == 'IMG':
                    name = FLAGS.imgs_dir+batch_sample[0].split('/')[-1]
                    center_image = imread(name)
                    name = FLAGS.imgs_dir+batch_sample[1].split('/')[-1]
                    left_image = imread(name)
                    name = FLAGS.imgs_dir+batch_sample[2].split('/')[-1]
                    right_image = imread(name)
                else:
                    center_image = imread(batch_sample[0])
                    left_image = imread(batch_sample[1])
                    right_image = imread(batch_sample[2])

                center_angle = float(batch_sample[3])

                # if(abs(center_angle) < 0.02):
                #     center_angle = 0
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(center_angle + 0.3)
                images.append(right_image)
                angles.append(center_angle - 0.3)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def main():
    samples = []
    # with open(FLAGS.csv_path) as csvfile:
    #     reader = csv.reader(csvfile)
    #     for line in reader:
    #         samples.append(line)
    #     samples.pop(0)

    with open(FLAGS.csv_path_augmented) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
    validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

    def resize(img):
        import tensorflow as tf
        img = tf.image.resize_images(img, (66, 200))
        return img

    model = Sequential([
            Cropping2D(cropping=((22,22), (0,0)), input_shape=(160,320,3)),
            Lambda(lambda x: (x / 255.0) - 0.5),
            Lambda(resize),
            Conv2D(24, (5, 5), padding='same', strides=(2,2), activation='relu'),
            SpatialDropout2D(0.2),
            Conv2D(36, (5, 5), padding='same', strides=(2,2), activation='elu'),
            SpatialDropout2D(0.2),
            Conv2D(48, (5, 5), padding='same', strides=(2,2), activation='elu'),
            SpatialDropout2D(0.2),
            Conv2D(64, (3, 3), padding='valid', activation='elu'),
            SpatialDropout2D(0.2),
            Conv2D(64, (3, 3), padding='same', activation='elu'),
            SpatialDropout2D(0.2),
            Flatten(),
            Dropout(0.5),
            Dense(100, activation='elu'),
            Dense(50, activation='elu'),
            Dense(10, activation='elu'),
            Dropout(0.5),
            Dense(1)
            ])

    model.compile(loss='mse', optimizer=Adam(lr=FLAGS.lrate))
    model.fit_generator(generator=train_generator,\
        steps_per_epoch=len(train_samples)//FLAGS.batch_size,\
        epochs=FLAGS.num_epochs,\
        validation_data=validation_generator,\
        validation_steps=len(validation_samples)//FLAGS.batch_size\
        )
    
    model.summary()
    model.save('model.h5')

if __name__ == '__main__':
    main()
