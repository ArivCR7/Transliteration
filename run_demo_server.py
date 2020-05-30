#!/usr/bin/env python3

import os

import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections

import torch
from torch.autograd import Variable
import crnn_pytorch.utils
from crnn_pytorch.dataset import resizeNormalize
from PIL import Image

import crnn_pytorch.models.crnn as crnn
import crnn_pytorch.params
import argparse

import tensorflow as tf
import model
from icdar import restore_rectangle
import lanms
from eval import resize_image, sort_poly, detect
from transliterator import Transliteration_EncoderDecoder

# Instantiates the device to be used as GPU/CPU based on availability
device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRANSLITERATION_MODELPATH = "./transliterationModel.pt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# code for transliteration
hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)

pad_char = '-PAD-'

hindi_alpha2index = {pad_char: 0}
for index, alpha in enumerate(hindi_alphabets):
    hindi_alpha2index[alpha] = index+1

eng_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

eng_alpha2index = {pad_char: 0}
for index, alpha in enumerate(eng_alphabets):
    eng_alpha2index[alpha] = index+1

#transliteratorModel = transliterator.Transliteration_EncoderDecoder(len(hindi_alpha2index), 256, len(eng_alpha2index), verbose=False)

transliteratorModel = torch.load(TRANSLITERATION_MODELPATH)
transliteratorModel.eval()

def word_rep(word, letter2index, device = 'cpu'):
    rep = torch.zeros(len(word)+1, 1, len(letter2index)).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        rep[letter_index][0][pos] = 1
    pad_pos = letter2index[pad_char]
    rep[letter_index+1][0][pad_pos] = 1
    return rep

def infer(net, hindi_word, max_chars, device='cpu'):
  net = net.to(device)
  input_ = word_rep(hindi_word, hindi_alpha2index, device)
  #gt_ = gt_rep(hin_word, hindi_alpha2ind, device)
  #hidden_ = net.init_hidden().to(device)
  out = net(input_, max_output_chars=max_chars, device=device)
  #outputs = torch.argmax(out)
  return(out)

def test(net, word, device = 'cpu'):
    net = net.eval().to(device)
    word = word.upper()
    outputs = infer(net, word, 30, device)
    #print(outputs)
    eng_output = ''
    for out in outputs:
        #index = torch.argmax(out).tolist()
        #print(list(eng_alpha2index.values()).index(torch.argmax(out)))
        eng_char = list(eng_alpha2index.keys())[list(eng_alpha2index.values()).index(torch.argmax(out))]
        index = list(eng_alpha2index.values()).index(torch.argmax(out))
        '''val, indices = out.topk(1)
        index = indices.tolist()[0][0]'''
        if index == 0:
            break
        #eng_char = eng_alphabets[index+1]
        eng_output += eng_char
    print(word + ' - ' + eng_output)
    return eng_output



@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret


@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
            'recognizedWords':[]
        }
        ret.update(get_host_info())
        return ret


    return predictor


### the webserver
from flask import Flask, request, render_template
import argparse


class Config:
    SAVE_DIR = 'static/results'


config = Config()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


def draw_illu(illu, rst):
    crnn_model_path = './crnn_pytorch/netCRNN_15_4000.pth'
    nclass = len(crnn_pytorch.params.alphabet) + 1
    model = crnn.CRNN(crnn_pytorch.params.imgH, crnn_pytorch.params.nc, nclass, crnn_pytorch.params.nh)
    if torch.cuda.is_available():
        model = model.cuda()
    # load model
    print('loading pretrained model from %s' % crnn_model_path)
    if crnn_pytorch.params.multi_gpu:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
    converter = crnn_pytorch.utils.strLabelConverter(crnn_pytorch.params.alphabet)

    transformer = resizeNormalize((100, 32))
    #image = Image.open(image_path).convert('L')
    recognizedWords = []
    recogEngWords = []
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
        print([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'], t['y2'], t['x3'], t['y3']])
        roi = illu[int(t['y0']):int(t['y2']), int(t['x0']):int(t['x2'])].copy()
        image = Image.fromarray(roi).convert('L')
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.LongTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        print('%-20s => %-20s' % (raw_pred, sim_pred))
        recogEng = test(transliteratorModel, sim_pred ,device=device_gpu)
        recognizedWords.append(sim_pred)
        recogEngWords.append(recogEng)

    rst['recognizedWords']=recognizedWords
    rst['recognizedEngWords']=recogEngWords

    return illu, recognizedWords


def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    illu, recognizedWords = draw_illu(img.copy(), rst)
    cv2.imwrite(output_path, illu)

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    print(recognizedWords)
    #rst['recognizedWords'] = recognizedWords
    return rst



checkpoint_path = './checkpoints/'


@app.route('/', methods=['POST'])
def index_post():
    global predictor
    import io
    bio = io.BytesIO()
    #request.files['image'].save(bio)
    img = request.files['image']
    img = cv2.imdecode(np.frombuffer(img.getvalue(), dtype='uint8'), 1)
    #print("Type of bio: ",type(bio.getvalue()))
    rst = get_predictor(checkpoint_path)(img)

    rst =   save_result(img, rst)
    return render_template('index.html', session_id=rst['session_id'])


def main():
    global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--checkpoint_path', default=checkpoint_path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', args.port)

if __name__ == '__main__':
    main()

