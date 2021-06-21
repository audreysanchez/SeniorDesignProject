import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
from scipy.misc import electrocardiogram
from scipy.io.wavfile import write
from scipy.signal import find_peaks
from picamera import PiCamera
from time import sleep
from keras import preprocessing
from keras.models import load_model
from glob import glob
import ezgmail
from twilio.rest import Client
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
# (silence)

#1) Microphone records, if frequency is of drone, it will pass to 2)

fs=44100
duration = 15  # seconds
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
print("Recording Audio")
sd.wait()
write('record.wav', fs, myrecording)
print("Audio recording complete , Play Audio")

sd.play(myrecording, fs)
sd.wait()
print("Play Audio Complete")

sample_rate, data = wav.read('sounds/record.wav' ) #sample_rate= samples/sec
print("sample_rate, data = wav.read('sounds/record.wav') #sample_rate= samples/sec")
print("sample rate", sample_rate)
print("data.shape ",data.shape)
data = data.mean(axis=1)
duration= data.shape[0] / sample_rate # number of samples / sampling rate
print("duration", duration)

N = int(sample_rate/2.0) #halt a second
f = fftfreq(N, 1.0/sample_rate)
t = np.linspace(0,0.5,N)
mask = (f > 0) * (f < 1000)
subdata = data[:N]
F = fft(subdata)

freqmax = abs(F[mask])

hz= np.argmax(freqmax)

print("The frequency is {} Hz".format(hz))


if (hz >=1000 and hz <= 5000 )or (hz >=100 and hz <= 500):
    print("Frequency detected is: DRONE")

    #2) Take picture of drone

    camera = PiCamera()
    camera.rotation=180
    camera.start_preview()
    sleep(5)
    camera.capture('/home/pi/Documents/train_test/test_data/images/image.jpeg')
    camera.stop_preview()

    sleep(10)

    #3) CNN Model will determine if picture is of a drone

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    class_names = ['DRONE', 'NO_DRONE']
    width = 150
    height = 150

    model = load_model('drone_detector.h5')

    base_path_img = './test_data/images'
    images = []
    path = os.path.join(base_path_img, '*.jpeg')
    for image_path in glob(path):
        img = preprocessing.image.load_img(image_path, target_size=(width, height))

    img_X = np.expand_dims(img, axis=0)

    predictions = model.predict(img_X)
    result = class_names[np.argmax(predictions)]

    print('The type predicted is: {}'.format(result))

    if(result == "DRONE"):
        #4.1)Send email to customer when drone detected
         ezgmail.send('customer123@gmail.com', 'Drone Detector Notification', 'Dear User: A DRONE was detected on your property, please take safe precautions', ["./test_data/images/image.jpeg"])
         #4.2)Send text message when drone detected
         accountSID = 'AC7f6696f5533bef467a95992df2ed3d1e'
         authToken = '595bc8aa845721d3f57da8d12ddd523d'
         twilioCli = Client(accountSID, authToken)
         myTwilioNumber = '+14842224954'
         cellPhone = '+19562219551'
         message = twilioCli.messages.create(body='Dear User - A DRONE was detected on your property- please take safe precautions.', from_=myTwilioNumber, to=cellPhone)
         media_url=["./test_data/images/image.jpeg"]
    else:
        print("No Drone on property")
else:
    print("Frequency detected is: NOT DRONE")