

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import time
import wandb
import scipy.io as sio

# from sklearn.preprocessing import OneHotEncoder
from model import *
from ofdm import *
from radio import *
from util import *
import copy
# these ones let us draw images in our notebook

flags = tf.compat.v1.app.flags
flags.DEFINE_string('save_dir', './output/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('nbits', 3, 'bits per symbol')
flags.DEFINE_integer('msg_length', 100800, 'Message Length of Dataset')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_integer('max_epoch_num', 5000, '')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('nfft', 64, 'Dropout rate TX conv block')
flags.DEFINE_integer('nsymbol', 7, 'Dropout rate TX conv block')
flags.DEFINE_integer('npilot', 8, 'Dropout rate TX dense block')
flags.DEFINE_integer('nguard', 8, 'Dropout rate RX conv block')
flags.DEFINE_integer('nfilter', 80, 'Dropout rate RX conv block')
flags.DEFINE_float('SNR', 30.0, '')
flags.DEFINE_float('SNR2', 30.0, '')
flags.DEFINE_integer('early_stop',400,'number of epoches for early stop')
flags.DEFINE_boolean('ofdm',True,'If add OFDM layer')
flags.DEFINE_string('pilot', 'lte', 'Pilot type: lte(default), block, comb, scattered')
flags.DEFINE_string('channel', 'Flat', 'AWGN or Rayleigh Channel: Flat, EPA, EVA, ETU')
flags.DEFINE_boolean('cp',True,'If include cyclic prefix')
flags.DEFINE_boolean('longcp',True,'Length of cyclic prefix: true 25%, false 7%')
flags.DEFINE_boolean('load_model',True,'Set True if run a test')
flags.DEFINE_float('split',1.0,'split factor for validation set, no split by default')
flags.DEFINE_string('token', 'OFDM_PA_Model_awgn_5dB_8QAM','Name of model to be saved')
flags.DEFINE_integer('opt', 3, '0: default equalizer, 1: NoCConv, 2: NoResidual, 3: DNN')
flags.DEFINE_boolean('mobile', False, 'If Doppler spread is turned on')
flags.DEFINE_float('init_learning', 0.001, '')
flags.DEFINE_boolean('test',True,'Test trained model')
FLAGS = flags.FLAGS

# Rewrite all the flags handle to tensorflow_v2 style without using tf.compat.v1.flags
def parse_args():
    '''Parses OFDM arguments.'''
    parser = argparse.ArgumentParser(description="Run OFDM.")

    # model params
    parser.add_argument('--nbits', nargs='?', default=3,
                        help='bits per symbol')
    parser.add_argument('--msg_length', nargs='?', default=100800,
                        help='Message Length of Dataset')
    parser.add_argument('--batch_size', nargs='?', default=512,
                        help='Batch Size')
    parser.add_argument('--max_epoch_num', nargs='?', default=5000,
                        help='Max Epoch Number')
    parser.add_argument('--seed', nargs='?', default=1,
                        help='random seed')
    parser.add_argument('--nfft', nargs='?', default=64,
                        help='FFT Size')
    parser.add_argument('--nsymbol', nargs='?', default=7,
                        help='Number of OFDM Symbols')
    parser.add_argument('--npilot', nargs='?', default=8,
                        help='Number of Pilot')
    parser.add_argument('--nguard', nargs='?', default=8,
                        help='Number of Guard')
    parser.add_argument('--nfilter', nargs='?', default=80,
                        help='Number of Filters in Conv Layer')
    parser.add_argument('--SNR', nargs='?', default=30.0,
                        help='SNR')
    parser.add_argument('--SNR2', nargs='?', default=30.0,
                        help='SNR')
    parser.add_argument('--early_stop', nargs='?', default=400,
                        help='number of epoches for early stop')
    parser.add_argument('--ofdm', nargs='?', default=True,
                        help='If add OFDM layer')
    parser.add_argument('--pilot', nargs='?', default='lte',
                        help='Pilot type: lte(default), block, comb, scattered')
    parser.add_argument('--channel', nargs='?', default='EPA',
                        help='AWGN or Rayleigh Channel: Flat, EPA, EVA, ETU')
    parser.add_argument('--cp', nargs='?', default=True,
                        help='If include cyclic prefix')
    parser.add_argument('--longcp', nargs='?', default=True,
                        help='Length of cyclic prefix: true 25%, false 7%')
    parser.add_argument('--load_model', nargs='?', default=True,
                        help='Set True if run a test')
    parser.add_argument('--

#wandb.init(project='DCNN-endtoend', entity='mhni',config = FLAGS)

class RayleighChanParallel:
    def __init__(self, parse_args, sample_rate=0.96e6, mobile=False, mix=False):
        self.cpu_count = mp.cpu_count()
        self.Fs = sample_rate
        self.flags = parse_args
        self.mobile = mobile
        self.mix = mix
        self.nfft = parse_args.nfft
        self.pool = mp.Pool(processes=self.cpu_count)
        self.create_objs()

    def create_objs(self):
        objs = []
        for i in range(self.cpu_count):
            fading_obj = rayleigh_chan_lte(self.flags, self.Fs, self.mobile, self.mix)
            objs.append(fading_obj)
        self.objs = objs

    def run(self, iq_tx_cmpx):
        n_fr, n_sym, n_sc = np.shape(iq_tx_cmpx)
        tx_signal_list = []
        chunk_size = np.ceil(n_fr/self.cpu_count).astype(int)
        for i in range(self.cpu_count):
            tx_chuck = iq_tx_cmpx[i*chunk_size:(i+1)*chunk_size, :, :]
            tx_signal_list.append(tx_chuck)

        results = [self.pool.apply(self.objs[i], args=(tx_signal_list[i],)) for i in range(self.cpu_count)]
        rx_signal = np.zeros([n_fr, n_sym, n_sc, 2], dtype=np.float)
        ch_ground = np.zeros([n_fr, n_sym, self.nfft], dtype=np.complex64)
        for i in range(self.cpu_count):
            rx_sig_sym, chan_sym = results[i]
            rx_signal[i*chunk_size:(i+1)*chunk_size, :, :, :] = rx_sig_sym
            ch_ground[i*chunk_size:(i+1)*chunk_size, :, :] = chan_sym
        return rx_signal, ch_ground



def main(argv):
    nbits = parse_args.nbits # BPSK: 2, QPSK, 4, 16QAM: 16
    m_order = np.exp2(nbits)
    ofdmobj = ofdm_tx(FLAGS)

    nfft = ofdmobj.K
    ofdm_pf = ofdmobj.nSymbol # 8
    # frame_size = nfft - ofdmobj.G - ofdmobj.P - ofdmobj.DC
    frame_size = ofdmobj.frame_size
    msg_length = parse_args.msg_length
    frame_cnt = parse_args.msg_length//parse_args.nsymbol
    #SNR = FLAGS.SNR # set 7dB Signal to Noise Ratio
    np.random.seed(parse_args.seed)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    layer_norm = tf.keras.layers.LayerNormalization(axis=1, center=False, scale=False)
    for n in range(16, 24): # determines how many models to use
        if n == 0:
            #PA_Model_Select = 'PA_Model_PAout_Steering_0_0deg_aligned'
            PA_Model_Select = 'PA_Model_m20_0dbm_aligned'
        elif n == 1:
            #PA_Model_Select = 'PA_Model_PAout_Steering_8_0deg_aligned'
            PA_Model_Select = 'PA_Model_m21_0dbm_aligned'
        elif n == 2:
            #PA_Model_Select = 'PA_Model_PAout_Steering_15_5deg_aligned'
            PA_Model_Select = 'PA_Model_m22_0dbm_aligned'
        elif n == 3:
            #PA_Model_Select = 'PA_Model_PAout_Steering_23_5deg_aligned'
            PA_Model_Select = 'PA_Model_m23_0dbm_aligned'
        elif n == 4:
            #PA_Model_Select = 'PA_Model_PAout_Steering_32_5deg_aligned'
            PA_Model_Select = 'PA_Model_m24_0dbm_aligned'
        elif n == 5:
            #PA_Model_Select = 'PA_Model_PAout_Steering_41_5deg_aligned'
            PA_Model_Select = 'PA_Model_m25_0dbm_aligned'
        elif n == 6:
            #PA_Model_Select = 'PA_Model_PAout_Steering_52_0deg_aligned'
            PA_Model_Select = 'PA_Model_m27_0dbm_aligned'
        elif n == 7:
            #PA_Model_Select = 'PA_Model_PAout_Steering_64_5deg_aligned'
            PA_Model_Select = 'PA_Model_m28_0dbm_aligned'
        elif n == 8:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m8_0deg_aligned'
            PA_Model_Select = 'PA_Model_m29_0dbm_aligned'
        elif n == 9:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m15_5deg_aligned'
            PA_Model_Select = 'PA_Model_m30_0dbm_aligned'
        elif n == 10:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m23_5deg_aligned'
            PA_Model_Select = 'PA_Model_80MHz_APA'
        elif n == 11:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m32_5deg_aligned'
            PA_Model_Select = 'PA_Model_90MHz_APA'
        elif n == 12:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m41_5deg_aligned'
            PA_Model_Select = 'PA_Model_100MHz_APA'
        elif n == 13:
            PA_Model_Select = 'PA_Model_PAout_Steering_m52_0deg_aligned'
        elif n == 14:
            PA_Model_Select = 'PA_Model_PAout_Steering_m64_5deg_aligned'
        elif n == 15:
            PA_Model_Select = 'PA_Model_PAout_Steering_m78_0deg_aligned'
        elif n == 16:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m8_0deg_aligned'
            PA_Model_Select = 'PA_Model__awgn_50dB'
        elif n == 17:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m15_5deg_aligned'
            PA_Model_Select = 'PA_Model__awgn_45dB'
        elif n == 18:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m23_5deg_aligned'
            PA_Model_Select = 'PA_Model__awgn_40dB'
        elif n == 19:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m32_5deg_aligned'
            PA_Model_Select = 'PA_Model__awgn_35dB'
        elif n == 20:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m41_5deg_aligned'
            PA_Model_Select = 'PA_Model__awgn_30dB'
        elif n == 21:
            PA_Model_Select = 'PA_Model__awgn_25dB'
        elif n == 22:
            PA_Model_Select = 'PA_Model__awgn_20dB'
        elif n == 23:
            PA_Model_Select = 'PA_Model__awgn_10dB'
        elif n == 24:
            PA_Model_Select = 'PA_Model__awgn_5dB'
        else:
            #PA_Model_Select = 'PA_Model_PAout_Steering_78_0deg_aligned'
            PA_Model_Select = 'PA_Model_100MHz_APA'
            # _100MHz_APA
        PA_Model_File = '/Users/mhni/Documents/AI-Rfsensor/dl_ofdm-master/PA_models/' + PA_Model_Select + '.mat'
        # PA_Model_File = '/Users/mhni/Documents/AI-Rfsensor/dl_ofdm-master/' + PA_Model_Select + '.mat'
        PA_Model_Data = sio.loadmat(PA_Model_File)

        MP_Weights = PA_Model_Data['PA_Model'].squeeze()
        Memory_Deep, Order = np.shape(MP_Weights)
        MP_Weights_Reshape = MP_Weights.T.reshape(-1)
        Model_Input_Power_Max = PA_Model_Data['Model_Input_Power_Max'].squeeze().tolist()

        PA_Model_Data = {'MP_Weights': MP_Weights_Reshape, 'Memory_Deep': Memory_Deep, 'Order': Order,
                         'Model_Input_Power_Max': Model_Input_Power_Max}

        if FLAGS.test:
            session = tf.compat.v1.Session(config=config)
            session.run(tf.compat.v1.global_variables_initializer())
            # if FLAGS.opt == 0:
            #     path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token + '_Equalizer_' + FLAGS.channel)
            # else:
            #     path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token + '_Equalizer%d_'%(FLAGS.opt)  + FLAGS.channel)
            #path_prefix_min = os.path.join(FLAGS.save_dir, FLAGS.token + '_Equalizer%d_'%(FLAGS.opt)  + FLAGS.channel)
            path_prefix_min = './output/OFDM_PA_Model_awgn_5dB_8QAM_Equalizer3_Flat'
            y, x, iq_receiver, outputs, total_loss, ber, berlin, conf_matrix, power_tx, noise_pwr, iq_rx, iq_tx, ce_mean, SNR = load_model_np(
                path_prefix_min, session)
            print("Final Test SNR: -10 to 30 dB")
            nfft = FLAGS.nfft
            nbits = FLAGS.nbits
            npilot = FLAGS.npilot  # last carrier as pilot
            nguard = FLAGS.nguard
            nsymbol = FLAGS.nsymbol
            DC = 2
            np.random.seed(int(time.time()))
            frame_size = ofdmobj.frame_size
            frame_cnt = 30000
            for test_chan in ['Flat']:
                df = pd.DataFrame(columns=['SNR', 'BER', 'Loss'])
                flagcp = copy.deepcopy(FLAGS)
                flagcp.channel = test_chan
                # fading = rayleigh_chan_lte(flagcp, ofdmobj.Fs, mobile=FLAGS.mobile)
                fading = RayleighChanParallel(flagcp, ofdmobj.Fs, mobile=FLAGS.mobile)
                print("Test in %s, mobile: %s" % (test_chan, FLAGS.mobile))
                for snr_t in range(-10, 31):
                    np.random.seed(int(time.time()) + snr_t)
                    test_ys = bit_source(nbits, frame_size, frame_cnt)
                    # iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_np(test_ys)
                    iq_tx_cmpx, test_xs, iq_pilot_tx = ofdmobj.ofdm_tx_frame_np(test_ys)
                    iq_tx_cmpx = PA_Model2(PA_In=iq_tx_cmpx, PA_Model_Data=PA_Model_Data)
                    test_xs, _ = fading.run(iq_tx_cmpx)
                    snr_test = snr_t * np.ones((frame_cnt, 1))
                    test_xs, pwr_noise_avg = AWGN_channel_np(test_xs, snr_test)
                    confmax, berl, pwr_tx, pwr_noise, test_loss, tx_sample, rx_sample = session.run(
                        [conf_matrix, berlin, power_tx, noise_pwr, ce_mean, iq_tx, iq_rx],
                        {x: test_xs, y: test_ys, SNR: snr_test})

                    print("SNR: %.2f, BER: %.8f, Loss: %f" % (snr_t, berl, test_loss))
                    print("Test Confusion Matrix: ")
                    print(str(confmax))
                    df = df.append({'SNR': snr_t, 'BER': berl, 'Loss': test_loss}, ignore_index=True)
                    # writer = tf.summary.create_file_writer('./graphs', graph = sess.graph)

                df = df.set_index('SNR')
                # csvfile = 'Test_DCCN_%s_test_chan_%s.csv'%(FLAGS.token + '_Equalizer_' + FLAGS.channel, test_chan)
                if FLAGS.mobile:
                    csvfile = 'ExampleTest_Test_DCCN_%s_test_chan_%s_mobile.csv' % (
                    FLAGS.token + '_Equalizer%d_' % (FLAGS.opt) + FLAGS.channel, test_chan)
                else:
                    csvfile = 'ExampleTest_Test_DCCN_%s_test_chan_%s.csv' % (
                    FLAGS.token + '_Equalizer%d_' % (FLAGS.opt) + FLAGS.channel + PA_Model_Select, test_chan)
                df.to_csv(csvfile)
            session.close()


if __name__ == "__main__":
    tf.compat.v1.app.run()