import numpy as np
import cv2
import os
import glob
import argparse



def remove_no_face_search(face_directory):
    file_list = os.listdir(face_directory)
    images = glob.glob(os.path.join(face_directory, '*.jpg'))
    number_of_files = len(file_list)

    face_image_count = 0
    no_face_count = 0
    for image in images:
        try:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        except:
            print(image)
            continue

        if (len(faces) != 0):
            face_image_count += 1
        else:
#             print('No face found in ' + image)
            os.remove(os.path.join(image))
            no_face_count += 1

    print('A total of ' + str(number_of_files) + ' was screened by haarcascade face detection.')
    print(str(face_image_count) + ' of images out of ' + str(number_of_files) + ' that have faces inside was found.')
    print(str(no_face_count) + ' of images out of ' + str(number_of_files) + ' that have no faces in side was found.')

def remove_face_search(no_face_directory):
    file_list = os.listdir(no_face_directory)
    images = glob.glob(os.path.join(no_face_directory, '*.jpg'))
    number_of_files = len(file_list)

    face_image_count = 0
    no_face_count = 0
    for image in images:
        try:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        except:
            print(image)
            continue

        if (len(faces) != 0):
            face_image_count += 1
            os.remove(os.path.join(image))
        else:
#             print('No face found in ' + image)
            no_face_count += 1

    print('A total of ' + str(number_of_files) + ' was screened by haarcascade face detection.')
    print(str(face_image_count) + ' of images out of ' + str(number_of_files) + ' that have faces inside was found.')
    print(str(no_face_count) + ' of images out of ' + str(number_of_files) + ' that have no faces in side was found.')

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier(frontal_face_detector_xml)
    eye_cascade = cv2.CascadeClassifier(eye_detector_xml)

    # Next two lines testing if face_cascade and eye_cascade are loaded
    # face_cascade.empty()
    # eye_cascade.empty()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--face_directory',
        type=str,
        help='face image path to check'
    )

    parser.add_argument(
        '--no_face_directory',
        type=str,
        help='no face image path to check'
    )
    parser.add_argument(
        '--frontal_face_detector_xml',
        type=str,
        default='/Users/wan/dev/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
        help='image path to check'
    )
    parser.add_argument(
        '--eye_detector_xml',
        type=str,
        default='/Users/wan/dev/opencv/data/haarcascades/haarcascade_eye.xml'
        help='image path to check'
    )


    args = parser.parse_args()

    remove_no_face_search(args.face_directory)
    remove_face_search(args.no_face_directory)
