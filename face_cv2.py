import numpy as np
import cv2
import os
import glob
import argparse

face_cascade = cv2.CascadeClassifier('/Users/wan/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/wan/dev/opencv/data/haarcascades/haarcascade_eye.xml')

# Next two lines testing if face_cascade and eye_cascade are loaded
# face_cascade.empty()
# eye_cascade.empty()

def face_search(directory):
    file_list = os.listdir(directory)
    images = glob.glob(os.path.join(directory, '*.jpg'))
    number_of_files = len(file_list)

    face_image_count = 0
    no_face_count = 0
    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces) != 0):
            face_image_count += 1
        else:
            print('No face found in ' + image)
            no_face_count += 1

    print('A total of ' + str(number_of_files) + 'was screened by haarcascade face detection.')
    print(str(face_image_count) + ' of images out of ' + str(number_of_files) + ' that have faces inside was found.')
    print(str(no_face_count) + ' of images out of ' + str(number_of_files) + ' that have no faces in side was found.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='image path to check'
    )

    args = parser.parse_args()

    face_search(args.directory)
