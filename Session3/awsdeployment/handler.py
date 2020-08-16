try:
    import unzip_requirements
except ImportError:
    pass

from PIL import Image
import dlib
import numpy as np
import faceBlendCommon as fbc

import boto3
import os
import io
import base64
from requests_toolbelt.multipart import decoder
import json
print("Imports End   here...")


# define env variables if there are not existing
#S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'sridevimodels'
#MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'shape_predictor_5_face_landmarks.data'

PREDICTOR_PATH = 'shape_predictor_5_face_landmarks.dat'

# Initialize the face detector
faceDetector = dlib.get_frontal_face_detector()

# Initialize the Landmark Predictor (a.k.a. shape predictor). The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

print("Face Detector and Landmark detector objects are created.....")

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_face_alignment(image_bytes):
    im = transform_image(image_bytes=image_bytes)

    # Detect Landmark
    points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    points = np.array(points)

    print('Landmarks detected')

    # Convert image to floating point in the range 0 to 1
    im = np.float32(im)/255.0

    # Specify the size of aligned face image. Compute the normalized image by using the similarity transform

    # Dimension of output Image
    h = im.shape[0] # 600
    w = im.shape[1] # 600

    # Normalize the image to output coordinates
    imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), im, points)
    imNorm = np.uint8(imNorm*255)
    print("Aligned image is made.....")

    # This is aligned image
    return imNorm

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def face_alignment_handler(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('content_type_header: ' + content_type_header)
        #print('Image content: ' + event['body'])
        print("Now Decoding.....")
        body = base64.b64decode(event['body'])
        print('Body Loaded')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print(f'MultipartDecoder processed')

        aligned_face = get_face_alignment(image_bytes=picture.content)
        print(f'Got the aligned face......')

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        print(filename)

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'file': filename.replace('"', ''), 'alignedFaceImg': img_to_base64(aligned_face)})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }

print("succesfully done.....")