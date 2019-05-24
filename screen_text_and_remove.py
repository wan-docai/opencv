import cv2 as cv
import math
import pandas as pd
import numpy as np
import os
import glob
import argparse

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def main(input):
    # Read and store arguments
    confThreshold = 0.9
    nmsThreshold = 0.4
    inpWidth = 320
    inpHeight = 320
    model = 'samples/dnn/frozen_east_text_detection.pb'

    # Load network
    net = cv.dnn.readNet(model)

    # Create a new named window
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")

    # Open a video file or an image file or a camera stream
#     df is file of image_path
#     for index, row in df.iterrows():
#         row[0] as input

    # cap = cv.VideoCapture(input if input else 0)
    cap = cv.imshow(input if input else 0)

    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        outs = net.forward(outNames)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        vertices_list = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i[0]])
            vertices_list.append(vertices)
        print(vertices_list)
#         total_area = 320*320 # generalize to args.height*args.width
        def calculatArea(vertices):
            total = 0
            for box in vertices:
                x0, y0 = box[0][0], box[0][1]
                x2, y2 = box[2][0], box[2][1]

                length = abs(x0 - x2)
                height = abs(y0 - y2)
                area = length * height

                total += area
            return total
        text_area = calculatArea(vertices_list)
#         print(type(text_area))
        total_area = 320*320
        percentage_of_text_area = float(text_area)/total_area * 100
        print(percentage_of_text_area)

        cv.destroyAllWindows()
        return percentage_of_text_area


def get_file_paths(dir_path, pattern = '*.jpg'):
    glob_pattern = os.path.join(dir_path, pattern)
    return glob.glob(glob_pattern)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='image path to check'
    )

    parser.add_argument(
        '--cutoff_percentage_of_area',
        type=str,
        default=10,
        help='what percentage of area as cut off to delete the image'
    )

    args = parser.parse_args()

    filepath = get_file_paths(args.directory)

    count = 0
    for file in filepath:
        percentage_of_text_area = main(file)
        if percentage_of_text_area >= args.cutoff_percentage_of_area:
            os.remove(file)
            count += 1

    print('Total files removed ' + str(count))
