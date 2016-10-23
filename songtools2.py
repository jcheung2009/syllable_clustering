#from sys import *
import scipy as sc
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy import io
from scipy.io import wavfile
from scipy import ndimage
from pylab import specgram, psd
from itertools import *
from scipy import signal
from array import array
import math
import evsonganaly

def impwav(a):
    """Imports a wave file as an array where a[1]
    is the sampling frequency and a[0] is the data"""
    out=[]
    wav = sc.io.wavfile.read(a)
    wf = np.array(wav[1]) - np.mean(wav[1])
    out=(wf, wav[0])
    return out

def impcbin(a):
    """Imports a cbin as an array where a[1] is the sampling freq and a[0] is the data"""
    file=open(a,'r')
    file=file.read()
    a=np.fromstring(file[1:-1],dtype=np.int16)
    out=[a,32000]
    return out

def impraw(a):
    """Imports a cbin as an array where a[1] is the sampling freq and a[0] is the data"""
    f = open(a,'r')
    data = f.read()
    data = np.fromstring(data[1:-1], dtype = np.int16)
    # data = np.divide(data,np.max(data))
    # plt.plot(data[100000:110000]); plt.show()
    out = (data, 40000)
    return out

def getsyls(a, min_length = 0.01, window = 2, threshold = 1):
    """takes a file read in with impwav and returns a list of sylables"""
    fs = a[1]
    fa = a;
    t = np.divide(np.arange(0,len(a[0])), float(fs))
    a=a[0]
    objs=findobject(smoothrect(fa, window, freq = fs), thresh = threshold)
    sylables=[x for x in [a[y] for y in objs[0]] if float(len(x))/fs > min_length]
    times=[x for x in [(t[y][0], t[y][-1]) for y in objs[0]] if (x[1]-x[0]) > min_length]
    return sylables, times

def filtersong(a):
    out=[]
    b=sc.signal.iirdesign(wp=0.04, ws=0.02, gpass=1, gstop=60,ftype='ellip')
    out.append(sc.signal.filtfilt(b[0],b[1],a[0]))
    out.append(a[1])
    return (out)

def is_sylable(a, fs = 32000):
    """Test whether the sound contained in array a is a sylable based on several criterian
    input:
    a = an array of samples
    fs = sampling rate
    output:
    true or False"""
    # parameters
    test = True
    max_length = 0.15 # seconds
    min_length = 0.02 # seconds
    if float(len(a))/fs < min_length: test = False
    elif float(len(a))/fs > max_length: test = False
    return test

def ffprofile_specgram(a, Fs = 32000, percent_boundry = 30, plot = False):
    # cacluclate spectrogram
    px, faxis, t = mlab.specgram(a, Fs=Fs, NFFT=1024, noverlap = 1000, window = mlab.window_hanning)
    # calculate overall fundamental frequency to stay within
    # ave_ff_freq = ffcalc_jk(a)[0]
    # ave_ff_freq = np.real(ave_ff_freq)
    # #px = np.log10(px)
    # f_idxs = np.arange(0,len(faxis),1)
    # f_idxs = f_idxs.astype(int)
    # f_idxs = f_idxs[(faxis > ave_ff_freq*float(1-float(percent_boundry)/100)) & (faxis <  ave_ff_freq*float(1+float(percent_boundry)/100))]
    # max_idxs = np.argmax(px[f_idxs,:], axis = 0)
    # max_idxs = f_idxs[max_idxs]
    # f_contour = faxis[max_idxs]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.specgram(sylable, Fs=Fs)
        # ax.plot(t, faxis,'k')
        # ax.plot([t[0], t[-1]], [ave_ff_freq, ave_ff_freq],'-b')
        # ax.plot([t[0], t[-1]], [ave_ff_freq*float(1+float(percent_boundry)/100), ave_ff_freq*float(1+float(percent_boundry)/100)],'b')
        # ax.plot([t[0], t[-1]], [ave_ff_freq*float(1-float(percent_boundry)/100), ave_ff_freq*float(1-float(percent_boundry)/100)],'b')
        plt.show()

    return  t, faxis

if __name__ == "__main__":
    filename = '/opt/data/pk31gr76/saline_8_9_2016/pk31gr76_090816_0801.4491.cbin'
    sylables,labels = evsonganaly.get_ev_sylables(evsonganaly.load_ev_file(filename, load_song=True))
    sylables = np.array(sylables)
    labels = np.array(labels)
    for ksyl,sylable in enumerate(sylables[labels=='a']):
        if is_sylable(sylable):
            print ksyl
            ffprofile_specgram(sylable,plot=True)


