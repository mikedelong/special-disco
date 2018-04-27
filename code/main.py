import logging
import time

import tensorflow as tf
import tensorlayer as tl

import data

start_time = time.time()

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = trainX.tolist()
    trainY = trainY.tolist()
    testX = testX.tolist()
    testY = testY.tolist()
    validX = validX.tolist()
    validY = validY.tolist()

    trainX = tl.prepro.remove_pad_sequences(trainX)
    trainY = tl.prepro.remove_pad_sequences(trainY)
    testX = tl.prepro.remove_pad_sequences(testX)
    testY = tl.prepro.remove_pad_sequences(testY)
    validX = tl.prepro.remove_pad_sequences(validX)
    validY = tl.prepro.remove_pad_sequences(validY)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
