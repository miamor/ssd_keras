## Convert weight
```
python caffe_to_keras.py weights/mobilenet_ssd mobilenet_deploy.prototxt weights/mobilenet_iter_73000.caffemodel --format hdf5
```
## Preprocess data
```
python get_data_from_XML.py
```   
Modify appropriate content 
## Train
```
python ssd_mobilenet.py train -o ./output/training_logs -w ./weights/mobilenet_ssd.hdf5
```
## Infer
```
python ssd_mobilenet.py infer -i ./test_images -o ./output/results -w ./output/training_logs/run1/checkpoint-276-2.7208.hdf5 -t 0.6
```
