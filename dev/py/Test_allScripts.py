

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
tf.compat.v1.app.flags.DEFINE_string('save_dir', './output/', 'directory where model graph and weights are saved')
tf.compat.v1.app.flags.DEFINE_integer('nbits', 3, 'bits per symbol')
tf.compat.v1.app.flags.DEFINE_integer('msg_length', 100800, 'Message Length of Dataset')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 512, '')
tf.compat.v1.app.flags.DEFINE_integer('max_epoch_num', 5000, '')
tf.compat.v1.app.flags.DEFINE_integer('seed', 1, 'random seed')
tf.compat.v1.app.flags.DEFINE_integer('nfft', 64, 'Dropout rate TX conv block')
tf.compat.v1.app.flags.DEFINE_integer('nsymbol', 7, 'Dropout rate TX conv block')
tf.compat.v1.app.flags.DEFINE_integer('npilot', 8, 'Dropout rate TX dense block')
tf.compat.v1.app.flags.DEFINE_integer('nguard', 8, 'Dropout rate RX conv block')
tf.compat.v1.app.flags.DEFINE_integer('nfilter', 80, 'Dropout rate RX conv block')
tf.compat.v1.app.flags.DEFINE_float('SNR', 30.0, '')
tf.compat.v1.app.flags.DEFINE_float('SNR2', 30.0, '')
tf.compat.v1.app.flags.DEFINE_integer('early_stop',400,'number of epoches for early stop')
tf.compat.v1.app.flags.DEFINE_boolean('ofdm',True,'If add OFDM layer')
tf.compat.v1.app.flags.DEFINE_string('pilot', 'lte', 'Pilot type: lte(default), block, comb, scattered')
tf.compat.v1.app.flags.DEFINE_string('channel', 'Flat', 'AWGN or Rayleigh Channel: Flat, EPA, EVA, ETU')
tf.compat.v1.app.flags.DEFINE_boolean('cp',True,'If include cyclic prefix')
tf.compat.v1.app.flags.DEFINE_boolean('longcp',True,'Length of cyclic prefix: true 25%, false 7%')
tf.compat.v1.app.flags.DEFINE_boolean('load_model',True,'Set True if run a test')
tf.compat.v1.app.flags.DEFINE_float('split',1.0,'split factor for validation set, no split by default')
tf.compat.v1.app.flags.DEFINE_string('token', 'OFDM_PA_Model_awgn_5dB_8QAM','Name of model to be saved')
tf.compat.v1.app.flags.DEFINE_integer('opt', 3, '0: default equalizer, 1: NoCConv, 2: NoResidual, 3: DNN')
tf.compat.v1.app.flags.DEFINE_boolean('mobile', False, 'If Doppler spread is turned on')
tf.compat.v1.app.flags.DEFINE_float('init_learning', 0.001, '')
tf.compat.v1.app.flags.DEFINE_boolean('test',True,'Test trained model')
FLAGS = tf.compat.v1.app.flags.FLAGS
flags = FLAGS
# Rewrite all the flags handle to tensorflow_v2 style without using tf.compat.v1.flags

#wandb.init(project='DCNN-endtoend', entity='mhni',config = FLAGS)

class RayleighChanParallel:
    def __init__(self, flags, sample_rate=0.96e6, mobile=False, mix=False):
        self.cpu_count = mp.cpu_count()
        self.Fs = sample_rate
        self.flags = flags
        self.mobile = mobile
        self.mix = mix
        self.nfft = flags.nfft
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
    nbits = flags.nbits # BPSK: 2, QPSK, 4, 16QAM: 16
    m_order = np.exp2(nbits)
    ofdmobj = ofdm_tx(FLAGS)

    nfft = ofdmobj.K
    ofdm_pf = ofdmobj.nSymbol # 8
    # frame_size = nfft - ofdmobj.G - ofdmobj.P - ofdmobj.DC
    frame_size = ofdmobj.frame_size
    msg_length = flags.msg_length
    frame_cnt = flags.msg_length//flags.nsymbol
    #SNR = FLAGS.SNR # set 7dB Signal to Noise Ratio
    np.random.seed(flags.seed)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    layer_norm = tf.keras.layers.LayerNormalization(axis=1, center=False, scale=False)
    for n in range(0,5): # determines how many models to use
        if n == 0:
            PA_Model_Select = 'PA_Model_BB1'
            #PA_Model_Select = 'PA_Model_m20_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_0_idq100_1'
        elif n == 1:
            #PA_Model_Select = 'PA_Model_PAout_Steering_8_0deg_aligned'
            #PA_Model_Select = 'PA_Model_m21_0dbm_aligned'
            PA_Model_Select = 'PA_Model_PA_highNon_20211003_n5dBm_modified'
            #PA_Model_Select = 'PA_Model_1_idq100_1'
        elif n == 2:
            #PA_Model_Select = 'PA_Model_PAout_Steering_15_5deg_aligned'
            #PA_Model_Select = 'PA_Model_m22_0dbm_aligned'
            PA_Model_Select = 'PA_Model_PA_highNon_20211103_n6dBm_modified'
            #PA_Model_Select = 'PA_Model_2_idq100_1'
        elif n == 3:
            #PA_Model_Select = 'PA_Model_PAout_Steering_23_5deg_aligned'
            #PA_Model_Select = 'PA_Model_m23_0dbm_aligned'
            PA_Model_Select = 'PA_Model_PA_highNon_20211103_n7dBm_modified'
            #PA_Model_Select = 'PA_Model_3_idq100_1'
        elif n == 4:
            #PA_Model_Select = 'PA_Model_PAout_Steering_32_5deg_aligned'
            PA_Model_Select = 'PA_Model_m24_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_4_idq100_1'
        elif n == 5:
            #PA_Model_Select = 'PA_Model_PAout_Steering_41_5deg_aligned'
            PA_Model_Select = 'PA_Model_m25_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_5_idq100_1'
        elif n == 6:
            #PA_Model_Select = 'PA_Model_PAout_Steering_52_0deg_aligned'
            PA_Model_Select = 'PA_Model_m27_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_6_idq100_1'
        elif n == 7:
            #PA_Model_Select = 'PA_Model_PAout_Steering_64_5deg_aligned'
            PA_Model_Select = 'PA_Model_m28_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_7_idq100_1'
        elif n == 8:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m8_0deg_aligned'
            PA_Model_Select = 'PA_Model_m29_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_8_idq100_1'
        elif n == 9:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m15_5deg_aligned'
            PA_Model_Select = 'PA_Model_m30_0dbm_aligned'
            #PA_Model_Select = 'PA_Model_9_idq100_1'
        elif n == 10:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m23_5deg_aligned'
            #PA_Model_Select = 'pa_model/PA_Model_80MHz_APA'
            PA_Model_Select = 'PA_Model_0_vg27_1'
        elif n == 11:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m32_5deg_aligned'
            #PA_Model_Select = 'pa_model/PA_Model_90MHz_APA'
            PA_Model_Select = 'PA_Model_1_vg27_1'
        elif n == 12:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m41_5deg_aligned'
            #PA_Model_Select = 'pa_model/PA_Model_100MHz_APA'
            PA_Model_Select = 'PA_Model_2_vg27_1'
        elif n == 13:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m52_0deg_aligned'
            PA_Model_Select = 'PA_Model_3_vg27_1'
        elif n == 14:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m64_5deg_aligned'
            PA_Model_Select = 'PA_Model_4_vg27_1'
        elif n == 15:
            #PA_Model_Select = 'PA_Model_PAout_Steering_m78_0deg_aligned'
            PA_Model_Select = 'PA_Model_5_vg27_1'
        elif n == 16:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m8_0deg_aligned'
            #PA_Model_Select = 'PA_Model__awgn_50dB'
            PA_Model_Select = 'PA_Model_6_vg27_1'
        elif n == 17:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m15_5deg_aligned'
            #PA_Model_Select = 'PA_Model__awgn_45dB'
            PA_Model_Select = 'PA_Model_7_vg27_1'
        elif n == 18:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m23_5deg_aligned'
            #PA_Model_Select = 'PA_Model__awgn_40dB'
            PA_Model_Select = 'PA_Model_8_vg27_1'
        elif n == 19:
            # PA_Model_Select = 'PA_Model_PAout_Steering_m32_5deg_aligned'
            #PA_Model_Select = 'PA_Model__awgn_35dB'
            PA_Model_Select = 'PA_Model_9_vg27_1'
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
        PA_Model_File = r'./PA_models/PA_models2/' + PA_Model_Select + '.mat'
        #PA_Model_File = r'./PA_models/' + PA_Model_Select + '.mat'
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
            path_prefix_min = './output/OFDM_PA_Model_awgn_5dB_8QAM_Equalizer3_EPA'
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
            for test_chan in ['EPA']:
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
                    iq_tx_cmpx = PA_Model(PA_In=iq_tx_cmpx, PA_Model_Data=PA_Model_Data)
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