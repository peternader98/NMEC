import cv2 as cv
import numpy as np
from keras import models

CLASS_NAMES = ['god nilos', 'khedive ismail', 'king fouad I', 'king thutmose III', 'mohamed ali', 'pen-hery the surveyor', 'pen menkh the governer of dendara', 'sphinx of king amenemhat III', 'the protective goddesses', 'writer']
PIXELS = (64, 64)

def recognise_image(files):
    model = models.load_model('utilities/NMEC.model')

    images = [cv.imread(file) for file in files]
    images = [cv.resize(image, PIXELS) for image in images]
    images = [cv.cvtColor(image, cv.COLOR_BGR2RGB) for image in images]

    predictions = [model.predict(np.array([image]) / 255.0) for image in images]
    indices = [np.argmax(prediction) for prediction in predictions]
    print([f'prediction is {CLASS_NAMES[index]}' for index in indices])

    return indices

def recognise_image_api(file):
    model = models.load_model('utilities/NMEC.model')

    image = cv.imread(file)
    image = cv.resize(image, PIXELS)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    prediction = model.predict(np.array([image]) / 255.0)
    index = np.argmax(prediction)
    print(f'prediction is {CLASS_NAMES[index]}')

    return index

def recognise_video(video):
    model = models.load_model('utilities/NMEC.model')

    description = ''

    frames = load_video(video)
    images = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]

    predictions = [model.predict(np.array([image]) / 255.0) for image in images]
    indices = [np.argmax(prediction) for prediction in predictions]
    prediction = set(indices)
    print([f'prediction is {CLASS_NAMES[index]}' for index in prediction])

    if len(list(prediction)) == 2:
        for index in list(prediction):
            description += CLASS_NAMES[index] + ' and '
        description = description[:-5]
    else:
        for index in list(prediction):
            description += CLASS_NAMES[index] + ', '
        description = description[:-2]
    print(description)

    return description

# Utilities to open video files using CV2

def load_video(path, max_frames=0, resize=PIXELS):
  cap = cv.VideoCapture(path[0])
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return frames