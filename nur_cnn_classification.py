from __future__ import division

from keras.layers import AveragePooling1D, Flatten, Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import objectives
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K


import numpy as np
from utilities import my_callbacks_11
from utilities import data_helper
import optparse
import sys
from keras import losses


def ranking_loss(y_true, y_pred):
    #ranking_loss without tree distance
    
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + y_pred - y_true, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

def compute_recall_ks(probas):
    recall_k = {}
    for group_size in [2, 5, 10]:
        recall_k[group_size] = {}
        print 'group_size: %d' % group_size
        for k in [1, 2, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall(probas, k, group_size)
                print 'recall@%d' % k, recall_k[group_size][k]
    return recall_k

def recall(probas, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in xrange(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        #p = np.random.permutation(len(batch))
        #indices = p[np.argpartition(batch[p], -k)[-k:]]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return n_correct / (len(probas) / test_size)

if __name__ == '__main__':
    # parse user input
    parser = optparse.OptionParser("%prog [options]")

    #file related options
    parser.add_option("-g", "--log-file",   dest="log_file", help="log file [default: %default]")
    parser.add_option("-d", "--data-dir",   dest="data_dir", help="directory containing list of train, test and dev file [default: %default]")
    parser.add_option("-m", "--model-dir",  dest="model_dir", help="directory to save the best models [default: %default]")

    parser.add_option("-t", "--max-length", dest="maxlen", type="int", help="maximul length (for fixed size input) [default: %default]") # input size
    parser.add_option("-f", "--nb_filter",         dest="nb_filter",     type="int",   help="nb of filter to be applied in convolution over words [default: %default]") 
    #parser.add_option("-r", "--filter_length",     dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]") 
    parser.add_option("-w", "--w_size",         dest="w_size", type="int",   help="window size length of neighborhood in words [default: %default]") 
    parser.add_option("-p", "--pool_length",       dest="pool_length",   type="int",   help="length for max pooling [default: %default]") 
    parser.add_option("-e", "--emb-size",          dest="emb_size",      type="int",   help="dimension of embedding [default: %default]") 
    parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int",   help="hidden layer size [default: %default]") 
    parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

    parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
    parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
    parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
    parser.add_option("-n", "--epochs",             dest="epochs", type="int", help="nb of epochs [default: %default]")

    parser.set_defaults(

        data_dir        = "./final_data/"
        ,log_file       = "log"
        ,model_dir      = "./saved_model/"

        ,learn_alg      = "rmsprop" # sgd, adagrad, rmsprop, adadelta, adam (default)
        ,loss           = "ranking_loss" # hinge, squared_hinge, binary_crossentropy (default)
        ,minibatch_size = 32
        ,dropout_ratio  = 0.5

        ,maxlen         = 1000
        ,epochs         = 30
        ,emb_size       = 100
        ,hidden_size    = 250
        ,nb_filter      = 150
        ,w_size         = 5 
        ,pool_length    = 6 
    )

    opts,args = parser.parse_args(sys.argv)

    print('Loading vocabs for the whole dataset...')
    vocabs, E = data_helper.init_vocab(opts.emb_size)

    print "--------------------------------------------------"

    print("loading entity-gird for pos and neg documents...")
    X_train_1, y_train_1  = data_helper.load_and_numberize_egrids_with_labels(filelist="./list.train.small.c", 
            maxlen=opts.maxlen, w_size=opts.w_size, vocabs=vocabs)

    X_dev_1, y_dev_1     = data_helper.load_and_numberize_egrids_with_labels(filelist="./list.dev.small.c", 
            maxlen=opts.maxlen, w_size=opts.w_size, vocabs=vocabs)

    X_test_1, y_test_1   = data_helper.load_and_numberize_egrids_with_labels(filelist="./list.test.small.c", 
            maxlen=opts.maxlen, w_size=opts.w_size, vocabs=vocabs)

    num_train = len(X_train_1)
    num_dev   = len(X_dev_1)
    num_test  = len(X_test_1)
    
    print num_train, num_dev, num_test
    #assign Y value
    #y_train_1 = [1] * num_train 
    #y_dev_1 = [1] * num_dev 
    #y_test_1 = [1] * num_test 

    print('.....................................')
    print("Num of traing pairs: " + str(num_train))
    print("Num of dev pairs: " + str(num_dev))
    print("Num of test pairs: " + str(num_test))
    #print("Num of permutation in train: " + str(opts.p_num)) 
    print("The maximum in length for CNN: " + str(opts.maxlen))
    print('.....................................')

    # the output is always 1??????
    #y_train_1 = np_utils.to_categorical(y_train_1, 2)
    #y_dev_1 = np_utils.to_categorical(y_dev_1, 2)
    #y_test_1 = np_utils.to_categorical(y_test_1, 2)

    #randomly shuffle the training data
    np.random.seed(113)
    np.random.shuffle(X_train_1)
    np.random.seed(113)
    np.random.shuffle(y_train_1)

    # first, define a CNN model for sequence of entities 
    sent_input = Input(shape=(opts.maxlen,), dtype='int32', name='sent_input')

    # embedding layer encodes the input into sequences of 300-dimenstional vectors. 
    x = Embedding(output_dim=opts.emb_size, weights=[E], input_dim=len(vocabs), input_length=opts.maxlen)(sent_input)

    # add a convolutiaon 1D layer
    x = Convolution1D(nb_filter=opts.nb_filter, filter_length = opts.w_size, border_mode='valid', 
            activation='relu', subsample_length=1)(x)

    # add max pooling layers
    
    x = MaxPooling1D(pool_length=opts.pool_length)(x)
    x = Dropout(opts.dropout_ratio)(x)
    x = Flatten()(x)
    x = Dropout(opts.dropout_ratio)(x)

    # add latent coherence score
    out_x = Dense(1, activation='linear')(x)

    final_model = Model(sent_input, out_x)

    #final_model.compile(loss='ranking_loss', optimizer='adam')
    final_model.compile(loss='binary_crossentropy', optimizer=opts.learn_alg)

    # setting callback
    histories = my_callbacks_11.Histories()

    #print(shared_cnn.summary())
    print(final_model.summary())

    print("------------------------------------------------")	
    
    
    model_name = opts.model_dir + "CNN_" + str(opts.dropout_ratio) + "_"+ str(opts.emb_size) + "_"+ str(opts.maxlen) + "_" \
    + str(opts.w_size) + "_" + str(opts.nb_filter) + "_" + str(opts.pool_length) + "_" + str(opts.minibatch_size)  
    
    print("Model name: " + model_name)

    print("Training model...")
    bestAcc = 0.0
    patience = 0 
    for ep in range(1,opts.epochs):
        
        final_model.fit(X_train_1, y_train_1, validation_data=(X_dev_1, y_dev_1), nb_epoch=1,
 					verbose=1, batch_size=opts.minibatch_size, callbacks=[histories])

        final_model.save(model_name + "_ep." + str(ep) + ".h5")

        curAcc =  histories.accs[0]
        if curAcc >= bestAcc:
            bestAcc = curAcc
            patience = 0
        else:
            patience = patience + 1

        #doing classify the test set
        y_pred = final_model.predict(X_test_1)        
        
        print("Perform on test set after Epoch: " + str(ep) + "...!")    
        
        recall_k = compute_recall_ks(y_pred)
        
        #stop the model whch patience = 8
        if patience > 10:
            print("Early stopping at epoch: "+ str(ep))
            break

    print("Model reachs the best performance on Dev set: " + str(bestAcc))
    print("Finish training and testing...")
    









