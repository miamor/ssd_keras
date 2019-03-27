import keras
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
#from SSD300.ssd300MobileNetV2Lite import SSD
from SSD300.ssd300MobileNet import SSD

from get_data_from_XML import XML_preprocessor
from generator import Generator

'''
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor',
               'p0', 'p1', 'p2']
'''
voc_classes = ['p0', 'p1', 'p2']
NUM_CLASSES = len(voc_classes) + 1
#NUM_CLASSES = 4
#batch_size = 128
#batch_size = 32
batch_size = 16
input_shape = (300, 300, 3)

#priorFile = './priorFiles/prior_boxes_ssd300MobileNetV2.pkl'
priorFile = './priorFiles/prior_boxes_ssd300MobileNet.pkl'
priors = pickle.load(open(priorFile, 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


def train(args):
    #weights = './weights/mobilenet_ssd.hdf5'
    #weights = './output/training_logs/run1/checkpoint-21-5.6296.hdf5'
    #weights = './output/training_logs/run1/checkpoint-37-3.4409.hdf5'

    #model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model = SSD(input_shape, num_classes=NUM_CLASSES, weights=args.weight)
    model.summary()
    model.load_weights(args.weight, by_name=False)
    print('Loaded weights to model.')

    loss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss
    model.compile(optimizer='Adadelta', loss=loss)
    #base_lr = 3e-4
    #optim = keras.optimizers.Adam(lr=base_lr)
    #model.compile(optimizer=optim, loss=loss)

    #data_parser = XML_preprocessor(data_path='/media/tunguyen/Others/Dataset/JapaneseCard/annotations/', num_classes=NUM_CLASSES-1)
    #keys = list(data_parser.data.keys())

    data_parser = pickle.load(open('japanese_data.pkl', 'rb'))
    keys = sorted(data_parser.keys())

    train_num = int(0.7 * len(keys))
    train_keys = keys[:train_num]
    val_keys = keys[train_num:]

    gen = Generator(gt=data_parser, bbox_util=bbox_util,
                    batch_size=batch_size, path_prefix='',
                    train_keys=train_keys, val_keys=val_keys, image_size=(300, 300))
    RUN = RUN + 1 if 'RUN' in locals() else 1

    LOG_DIR = './{}/run{}'.format(args.output_dir, RUN)
    LOG_FILE_PATH = LOG_DIR + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

    EPOCHS = 8000

    tensorboard = TensorBoard(log_dir=LOG_DIR, write_images=True)
    checkpoint = ModelCheckpoint(
        filepath=LOG_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    history = model.fit_generator(generator=gen.generate(True), steps_per_epoch=(gen.train_batches//gen.batch_size),
                                  validation_data=gen.generate(False), validation_steps=(gen.train_batches//gen.batch_size),
                                  epochs=EPOCHS, verbose=1, callbacks=[tensorboard, checkpoint])


def infer(args):
    #model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model = SSD(input_shape, num_classes=NUM_CLASSES, weights=args.weight)

    loss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss
    model.compile(optimizer='Adadelta', loss=loss)
    model.summary()
    #model.load_weights('./weights/MobileNetSSD300weights_voc_2007.hdf5', by_name=True)
    model.load_weights(args.weight, by_name=False)


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
    for j, img in enumerate(images):
        # Parse the outputs.
        det_label = results[j][:, 0]
        det_conf = results[j][:, 1]
        det_xmin = results[j][:, 2]
        det_ymin = results[j][:, 3]
        det_xmax = results[j][:, 4]
        det_ymax = results[j][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= args.thresh]

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
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            #cv2.putText(img, display_txt, (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[label], 2)
            cv2.putText(img, display_txt, (xmin+10, ymax-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        out_path = os.path.join(args.output_dir, imgs_name[j])
        cv2.imwrite(out_path, img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train')
    inferParser = subparsers.add_parser('infer')

    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument('-w', '--weight', action='store',
                        type=str, default='./weights/mobilenet_ssd.hdf5')
    trainParser.add_argument('-o', '--output_dir', action='store',
                        type=str, default='./output/training_logs')

    inferParser.add_argument('-i', '--input_dir', action='store',
                        type=str, default='./test_images')
    inferParser.add_argument('-w', '--weight', action='store',
                        type=str, default='./weights/mobilenet_ssd.hdf5')
    inferParser.add_argument('-t', '--thresh', action='store',
                        type=float, default='0.6')
    inferParser.add_argument('-o', '--output_dir', action='store',
                        type=str, default='./output/results')

    args = parser.parse_args()

    infer(args)
