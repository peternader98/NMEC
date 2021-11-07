from tensorflow.keras import layers, models
from keras.constraints import MaxNorm
from tensorflow.keras.preprocessing import image

# get images dataset into training and testing

PIXELS = 64
train = image.ImageDataGenerator(rescale= 1/255)
test = image.ImageDataGenerator(rescale= 1/255)

training_images = train.flow_from_directory('./datasets/training', target_size=(PIXELS, PIXELS), class_mode='categorical')
testing_images = test.flow_from_directory('./datasets/testing', target_size=(PIXELS, PIXELS), class_mode='categorical')

# print(training_images.class_indices)
# print(training_images.classes)

# use convolution(conv2D), pooling(MaxPooling2D), flatten and dense
# (units = , filter = (), activation = , input_shape = (weidth, hight, channle_no))

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(PIXELS, PIXELS, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu', kernel_constraint=MaxNorm(3)))

model.add(layers.Dense(128, activation='relu', kernel_constraint=MaxNorm(3)))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, validation_data=testing_images, epochs=30, batch_size=64)

loss, accuracy = model.evaluate(testing_images)
print(f'Loss : {loss} , Accuracy : {accuracy}')

model.save('utilities/NMEC.model')