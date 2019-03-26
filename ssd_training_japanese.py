from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import pickle
import keras

from SSD300.ssd_v2 import SSD300v2
from SSD300.ssd_training import MultiboxLoss
from SSD300.ssd_utils import BBoxUtility
from SSD300.ssd300MobileNetV2Lite import SSD

from get_data_from_XML import XML_preprocessor
from generator import Generator

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor',
               'p0', 'p1', 'p2']
NUM_CLASSES = len(voc_classes) + 1
#NUM_CLASSES = 4
batch_size = 128
input_shape = (300, 300, 3)

#model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
model = SSD(input_shape, num_classes=NUM_CLASSES)
model.summary()
#model.load_weights('./weights/MobileNetSSD300weights_voc_2007.hdf5', by_name=True)

loss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss
model.compile(optimizer='Adadelta', loss=loss)
#base_lr = 3e-4
#optim = keras.optimizers.Adam(lr=base_lr)
#model.compile(optimizer=optim, loss=loss)

#priors = pickle.load(open('./priorFiles/prior_boxes_ssd300.pkl', 'rb'))
priors = pickle.load(open('./priorFiles/prior_boxes_ssd300MobileNetV2.pkl', 'rb'))
print(priors.shape)
print(priors)
bbox_util = BBoxUtility(NUM_CLASSES, priors)

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

LOG_DIR = './output/training_logs/run{}'.format(RUN)
LOG_FILE_PATH = LOG_DIR + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

EPOCHS = 32

tensorboard = TensorBoard(log_dir=LOG_DIR, write_images=True)
checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit_generator(generator=gen.generate(True), steps_per_epoch=int(gen.train_batches / 4),
                              validation_data=gen.generate(False), validation_steps=int(gen.val_batches / 4),
                              epochs=EPOCHS, verbose=1, callbacks=[tensorboard, checkpoint])
'''
history = model.fit_generator(generator=gen.generate(True), steps_per_epoch=(gen.train_batches//gen.batch_size),
                              validation_data=gen.generate(False), validation_steps=(gen.val_batches//gen.batch_size + 1),
                              epochs=EPOCHS, verbose=1, callbacks=[tensorboard, checkpoint, early_stopping])
'''