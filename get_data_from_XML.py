import numpy as np
import os
from xml.etree import ElementTree
import pickle

class XML_preprocessor(object):

    def __init__(self, data_path, num_classes=3):
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        #one_hot_vector[int(name)] = 1
        if name == 'p0':
            one_hot_vector[0] = 1
        elif name == 'p1':
            one_hot_vector[1] = 1
        elif name == 'p2':
            one_hot_vector[2] = 1

        return one_hot_vector



voc_classes = ['p0', 'p1', 'p2']
NUM_CLASSES = len(voc_classes)

#data = XML_preprocessor('/media/tunguyen/Others/Dataset/JapaneseCard/annotations/', NUM_CLASSES).data
data = XML_preprocessor('data/annotations/', NUM_CLASSES).data
pickle.dump(data,open('./japanese_data.pkl','wb'))
