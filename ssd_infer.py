from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pickle
import os
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

from SSD300.ssd_v2 import SSD300v2
from SSD300.ssd_training import MultiboxLoss
from SSD300.ssd_utils import BBoxUtility
from SSD300.ssd300MobileNet import SSD

from get_data_from_XML import XML_preprocessor
from generator import Generator

'''
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor',
               'p0', 'p1', 'p2']
'''
voc_classes = ['p0', 'p1', 'p2']
NUM_CLASSES = len(voc_classes) + 1
input_shape = (300, 300, 3)


def infer(args):
    thresh = args.thresh
    weights = args.weight

    #model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model = SSD(input_shape, num_classes=NUM_CLASSES, weights=weights)

    loss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss
    model.compile(optimizer='Adadelta', loss=loss)
    # model.summary()

    #model.load_weights('./weights/MobileNetSSD300weights_voc_2007.hdf5', by_name=True)
    model.load_weights(weights, by_name=False)

    #priors = pickle.load(open('./priorFiles/prior_boxes_ssd300.pkl', 'rb'))
    priors = pickle.load(
        open('./priorFiles/prior_boxes_ssd300MobileNet.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    inputs = []
    images = []
    imgs_name = []
    for f in os.listdir(args.input_dir):
        file_path = "{}/{}".format(args.input_dir, f)
        imgs_name.append(f)
        img = image.load_img(file_path, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(file_path))
        inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))

    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    colors = [None] * NUM_CLASSES
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            #label_name = voc_classes[label - 1]
            label_name = label - 1
            display_txt = '{:0.2f}, {}'.format(score, label_name)

            if colors[label] is None:
                colors[label] = tuple(
                    (np.random.choice(range(256), size=3)).reshape(1, -1)[0])

            #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[label], 1)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            #cv2.putText(img, display_txt, (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[label], 2)
            cv2.putText(img, display_txt, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        out_path = os.path.join(args.output_dir, imgs_name[i])
        cv2.imwrite(out_path, img)


def main(args):
    infer(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', action='store',
                        type=str, default='./test_images')
    parser.add_argument('-w', '--weight', action='store',
                        type=str, default='./weights/mobilenet_ssd.hdf5')
    parser.add_argument('-t', '--thresh', action='store',
                        type=float, default='0.6')
    parser.add_argument('-o', '--output_dir', action='store',
                        type=str, default='./output/results')

    args = parser.parse_args()

    infer(args)
