'''
A tool to convert `.caffemodel` weights to Keras-compatible HDF5 files or to export them to a simpler Python dictionary structure for further processing.

Copyright (C) 2018 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
os.environ['GLOG_minloglevel'] = '2' # Prevents Caffe from printing sooo much stuff to the console.
import caffe
os.environ['GLOG_minloglevel'] = '0'
import numpy as np
import warnings
import argparse
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You can export the weights to an HDF5 file only.")
try:
    import h5py
except ImportError:
    warning.warn("'h5py' module is missing. You can export the weights to a pickle file only.")

def convert_caffemodel_to_keras(output_filename,
                                prototxt_filename,
                                caffemodel_filename,
                                include_layers_without_weights=False,
                                include_unknown_layer_types=True,
                                keras_backend='tf',
                                verbose=True):
    if keras_backend != 'tf':
        raise ValueError("Only the TensorFlow backend is supported at the moment.")

    caffe_weights_list = convert_caffemodel_to_dict(prototxt_filename,
                                                    caffemodel_filename,
                                                    out_path=None,
                                                    verbose=False)

    out_name = '{}.hdf5'.format(output_filename)
    print(out_name)
    out = h5py.File(out_name, 'a')

    layer_names = []

    counter_unknown = 0
    counter_no_weights = 0

    iterator = iter(range(len(caffe_weights_list)))

    for i in iterator:
        layer = caffe_weights_list[i]
        layer_name = layer['name']+'_keras'
        layer_type = layer['type']
        if (len(layer['weights']) > 0) or include_layers_without_weights: # Check whether this is a layer that contains weights.
            if layer_type in {'Convolution', 'Deconvolution', 'InnerProduct'}: 
                kernel = layer['weights'][0]
                if layer_type in {'Convolution', 'Deconvolution'}:
                    kernel = np.transpose(kernel, (2, 3, 1, 0))
                if layer_type == 'InnerProduct':
                    kernel = np.transpose(kernel, (1, 0))
                weight_names = ['kernel']
                if (len(layer['weights']) > 1):
                    bias = layer['weights'][1]
                    weight_names.append('bias')
                extended_weight_names = np.array(['{}/{}:0'.format(layer_name, weight_names[k]).encode() for k in range(len(weight_names))])
                group = out.create_group(layer_name)
                group.attrs.create(name='weight_names', data=extended_weight_names)
                subgroup = group.create_group(layer_name)
                subgroup.create_dataset(name='{}:0'.format(weight_names[0]), data=kernel)
                if (len(layer['weights']) > 1):
                    subgroup.create_dataset(name='{}:0'.format(weight_names[1]), data=bias)
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Converted weights for layer '{}' of type '{}'".format(layer_name, layer_type))
            elif layer['type'] == 'BatchNorm': 
                weights = []
                weight_names = []
                mean = layer['weights'][0]
                variance = layer['weights'][1]
                next_layer = caffe_weights_list[i + 1]
                if next_layer['type'] == 'Scale':
                    gamma = next_layer['weights'][0]
                    weights.append(gamma)
                    weight_names.append('gamma')
                    if (len(next_layer['weights']) == 1):
                        warnings.warn("This 'Scale' layer follows a 'BatchNorm' layer and is expected to have a bias, but it doesn't. Make sure to set `center = False` in the respective Keras batch normalization layer.")
                    else:
                        beta = next_layer['weights'][1]
                        weights.append(beta)
                        weight_names.append('beta')
                    # Increment the iterator by one since we need to skip the subsequent 'Scale' layer after we're done here.
                    next(iterator)
                else:
                    warnings.warn("No 'Scale' layer after 'BatchNorm' layer. Make sure to set `scale = False` and `center = False` in the respective Keras batch normalization layer.")
                weights.append(mean)
                weights.append(variance)
                weight_names.append('moving_mean') 
                weight_names.append('moving_variance') 
                extended_weight_names = np.array(['{}/{}:0'.format(layer_name, weight_names[k]).encode() for k in range(len(weight_names))])
                group = out.create_group(layer_name)
                group.attrs.create(name='weight_names', data=extended_weight_names)
                subgroup = group.create_group(layer_name)
                for j in range(len(weights)):
                    subgroup.create_dataset(name='{}:0'.format(weight_names[j]), data=weights[j])
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Converted weights for layer '{}' of type '{}'".format(layer_name, layer_type))
            elif (len(layer['weights']) > 0) and include_unknown_layer_types: # For all other (unsupported) layer types...
                weight_names = ['weights_{}'.format(i) for i in range(len(layer['weights']))]
                extended_weight_names = np.array(['{}/{}:0'.format(layer_name, weight_names[k]).encode() for k in range(len(weight_names))])
                group = out.create_group(layer_name)
                group.attrs.create(name='weight_names', data=extended_weight_names)
                subgroup = group.create_group(layer_name)
                for j in range(len(layer['weights'])):
                    subgroup.create_dataset(name='{}:0'.format(weight_names[j]), data=layer['weights'][j])
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Converted weights for layer '{}' of unknown type '{}'".format(layer_name, layer_type))
                counter_unknown += 1
            elif (len(layer['weights']) == 0):
                group = out.create_group(layer_name)
                group.attrs.create(name='weight_names', data=np.array([]))
                subgroup = group.create_group(layer_name)
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Processed layer '{}' of type '{}' which doesn't have any weights".format(layer_name, layer_type))
                counter_no_weights += 1
            elif verbose:
                print("Skipped layer '{}' of unknown type '{}'".format(layer_name, layer_type))
        elif verbose:
            print("Skipped layer '{}' of type '{}' because it doesn't have any weights".format(layer_name, layer_type))
    out.attrs.create(name='layer_names', data=np.array(layer_names))
    out.attrs.create(name='backend', data=b'tensorflow')
    out.attrs.create(name='keras_version', data=b'2.0.8')
    # We're done, close the output file.
    out.close()
    print("Weight conversion complete.")
    if verbose:
        print("{} \t layers were processed, out of which:".format(len(layer_names)))
        print("{} \t were of an unknown layer type".format(counter_unknown))
        print("{} \t did not have any weights".format(counter_no_weights))
    print('File saved as {}'.format(out_name))

def convert_caffemodel_to_dict(prototxt_filename,
                               caffemodel_filename,
                               out_path=None,
                               verbose=False):
    net = caffe.Net(prototxt_filename, 1, weights=caffemodel_filename)
    layer_list = []
    for li in range(len(net.layers)): # For each layer in the net...
        layer = {}
        layer['name'] = net._layer_names[li]
        layer['type'] = net.layers[li].type
        layer['weights'] = [net.layers[li].blobs[bi].data[...]
                            for bi in range(len(net.layers[li].blobs))]
        layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                             for bi in list(net._bottom_ids(li))]
        layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                          for bi in list(net._top_ids(li))]
        layer_list.append(layer)
        if verbose:
            print("Processed layer '{}' of type '{}'".format(layer['name'], layer['type']))

    del net

    if verbose:
        print("Weight extraction complete. Processed {} layers.".format(len(layer_list)))

    if not (out_path is None):
        out_name = '{}.pkl'.format(out_path)
        with open(out_name, 'wb') as f:
            pickle.dump(layer_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved as {}.'.format(out_name))

    return layer_list

def main(argv):
    if argv.format == 'hdf5':
        convert_caffemodel_to_keras(output_filename=argv.out_file,
                                    prototxt_filename=argv.prototxt,
                                    caffemodel_filename=argv.caffemodel,
                                    include_layers_without_weights=argv.include_non_weight,
                                    include_unknown_layer_types=not(argv.skip_unknown),
                                    keras_backend=argv.backend,
                                    verbose=argv.verbose)
    elif argv.format == 'pickle':
        _ = convert_caffemodel_to_dict(prototxt_filename=argv.prototxt,
                                       caffemodel_filename=argv.caffemodel,
                                       out_path=argv.out_file,
                                       verbose=argv.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_file', action='store', type=str, help='The output filename as the full path, but excluding the file extension.')
    parser.add_argument('prototxt', action='store', type=str, help='The filename (full path including file extension) of the `.prototxt` file that defines the Caffe model. ')
    parser.add_argument('caffemodel', action='store', type=str, help='The filename (full path including file extension) of the `.caffemodel` file that contains the weights for the Caffe model.')
    parser.add_argument('-f', '--format', action='store', type=str, default='hdf5', choices={'hdf5', 'pickle'}, help="To which format to export the weights.")
    parser.add_argument('-n', '--include_non_weight', action='store_true', default=False, help="This option is only relevant if the output format is HDF5. Include layers that have no weights ")
    parser.add_argument('-u', '--skip_unknown', action='store_true', default=False, help="This option is only relevant if the output format is HDF5. Skip layer types that are unknown to the ")
    parser.add_argument('-b', '--backend', action='store', type=str, default='tf', choices={'tf'}, help="This option is only relevant if the output format is HDF5. For which Keras backend to convert the weights. ")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Prints out the conversion status for every layer.")

    args = parser.parse_args()

    main(args)
