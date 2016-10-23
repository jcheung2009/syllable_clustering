import sys as sys
import os as os
import scipy as sc
from scipy import io
from scipy.io import wavfile
from scipy import ndimage
from scipy import signal
from pylab import specgram, psd
#from matplotlib.mlab import psd
import numpy as np
import sklearn as skl
from sklearn import cluster
from sklearn import metrics
from scipy import spatial
from scipy.stats.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import decomposition
import random as rnd
import cPickle as pk
import time as tm
#import entropy_estimators as ee


"""This script creates a gaussian mixture model for syllables in song data. It automatically estimates the number of discrete syllables in the song data and creates classification
labels for the data.

usage: python sylcluster_gaussian_mixture_expectation_max_dbscan_kfoldcv_noplot_50x.py <folder with wav files>
example: python sylcluster_gaussian_mixture_expectation_max_dbscan_kfoldcv_noplot_50x.py bk1bk6/

It automatically segments the first 3000 syllables from the song data and uses these as a data set.

It creates several output files:
psds.pk: a python pickle file which contains the power spectral densities for the syllables analylzed in the order in which they were segmented.
labs.pk: a python pickle file which contains the labels for the syllables analyzed.  They are in the same order as in psds.pk.
scores.pk: a python pickle file which contains the likelihoods for the syllables analyzed.  They are in the same order as the labs and psds.
seged_dict.pk: A python pickle file which contains the segmentation objects for all syllables organized into songs.  Allows reassignment of other data to song strings.
syls.pk: a python pickle file containing the raw segmented syllable data.

It creates a summary file with the following information:
Syllable number predicted by BIC
Syllable number predicted by likelihhod
The number of syllables analized
the length of the reference set


"""

#functions

def findobject(file):
 value=(np.average(file))/4
 thresh=threshold(file,value)
 label=(sc.ndimage.label(thresh)[0])
 objs=sc.ndimage.find_objects(label)
 return(objs)

def impwav(a):
 """Imports a wave file as an array where a[1]
 is the sampling frequency and a[0] is the data"""
 out=[]
 try:
  wav = sc.io.wavfile.read(a)
 except IOError: return []
 out=[wav[1],wav[0]]
 return out

def impcbin(a):
    """Imports a cbin as an array where a[1] is the sampling freq and a[0] is the data"""
    file=open(a,'r')
    file=file.read()
    a=np.fromstring(file[1:-1],dtype=np.int16)
    out=[a,32000]
    return out

def outwave(filename,array):
 """Exports a numpy array (or just a regular python array)
 as a wave file. It expects an array of the following format: (speed,data)"""
 sc.io.wavfile.write(filename,array[0],array[1])

def filtersong(a):
 out=[]
 b=sc.signal.iirdesign(wp=0.04, ws=0.02, gpass=1, gstop=60,ftype='ellip')
 out.append(sc.signal.filtfilt(b[0],b[1],a[0]))
 out.append(a[1])
 return (out)

def smoothrect(a,window=None,freq=None):
 """smooths and rectifies a song.  Expects a file format (data,samplerate).
    If you don't enter a smoothing window size it will use 2ms as a default."""
 if freq== None: freq=32000
 if window == None: window=2
 le=int(round(freq*window/1000))
 h=np.ones(le)/le
 smooth= np.convolve(h,abs(a))
 offset = int(round((len(smooth)-len(a))/2))
 smooth=smooth[(1+offset):(len(a)+offset)]
 return smooth

def getsyls(a):
 """takes a file red in with impwav and returns a list of sylables"""
 fa=filtersong(a)
 a=a[0]
 objs=findobject(smoothrect(fa[0],20))
 sylables=[x for x in [a[y] for y in objs] if len(x)>(10*32)]
 return sylables,objs

def threshold(a,thresh=None):
 """Returns a thresholded array of the same length as input
 with everything below a specific threshold set to 0.
 By default threshold is sigma."""
 if thresh==None: thresh = sc.std(a)
 out=np.where(abs(a)>thresh,a,np.zeros(a.shape))
 return out

def maxnorm(a):
 """normalizes a string by it's max"""
 a=a/max(a)
 return a

def chunks(l, n):
 """ Yield successive n-sized chunks from l.
 """
 out=[]
 lens=(len(l)/n)
 for i in range(0, lens):
  out.append(l[i*n:i*n+n])
 out.append(l[lens*n:])
 #print len(out)
 #leng=max([(lens.count(x),x) for x in set(lens)])[1]
 #out=[x for x in out if len(x)==leng]
 #print len(out)
 return out[:-1]

def strat_chunks(l,n):
 """ Yield successive n-sized chunks from l.
 """
 out=[]
 l=sorted(l,key=lambda x:sum(x))
 for i in range(0, n):
  out.append(l[i::n])
 return out

def EMofgmmcluster(array_of_syls,syll_types,all_labels,plot=False):
 """takes an array of segmented sylables and clusters them by
 taking psds (welch method) and fitting a mixture model"""
 nfft=2**14
 fs=32000
 segstart=int(round(600/(fs/float(nfft))))
 segend=int(round(16000/(fs/float(nfft))))
 welchpsds=[psd(x,NFFT=nfft,Fs=fs) for x in array_of_syls]
 segedpsds=[norm(x[0][segstart:segend]) for x in welchpsds]
 for x in range(len(array_of_syls)): #appends duration of syllable in seconds at end of PSD vector
  segedpsds[x] = np.append(segedpsds[x],len(array_of_syls[x])/float(fs))
 models = len(syll_types)+1
 #pca=skl.decomposition.PCA(n_components=16000)
 #segedpsds=pca.fit_transform(segedpsds)

 weights = np.ones(len(segedpsds[0]))
 weights[-1] = 2
 mean_param = []
 for syll_type in syll_types:
     syll_ind = [i for i,label in enumerate(all_labels) if label==syll_type]
     syll_psds = [segedpsds[i] for i in syll_ind]
     mean_param.append(np.mean(syll_psds,axis=0))
 # s = sc.spatial.distance.cdist(segedpsds,basis_set,'wminkowski',p=2,w=weights)
 # d=sc.spatial.distance.cdist(mean_param,basis_set,'wminkowski',p=2,w=weights)
 gmm = mixture.GMM(n_components=models,n_iter=100000,n_init=5,covariance_type='full')
 gmm.means_ = np.array(mean_param)
 gmm.fit(segedpsds)

 s=d.T[:1000].T
 loglikelihood=[]
 liklis=[]
 bic=[]
 aic=[]
 k=3
 reps=1
 for x in models:
  errs=[]
  bics=[]
  aics=[]
  for m in range(reps):
   ss=chunks(s,len(s)/k)
   for y in range(len(ss)):
    testset=ss[y]
    trainset=[]
    [trainset.extend(ss[n]) for n in [z for z in range(len(ss)) if z!=y]]
    gmm = mixture.GMM(n_components=x, n_iter=100000,n_init=5, covariance_type='full')
    gmm.fit(np.array(trainset))
    likeli=sum(gmm.score(np.array(testset)))
    errs.append(likeli)
    trainset=np.array(trainset)
    bics.append(gmm.bic(trainset))
    aics.append(gmm.aic(trainset))
  loglikelihood.append(np.mean(errs))
  liklis.append(errs)
  bic.append(bics)
  aic.append(aics)
 bic=[np.mean(x) for x in bic]
 pvals=[sc.stats.mannwhitneyu(liklis[x],liklis[x+1]) for x in range(len(liklis)-1)]
 mins=[x for x in range(len(pvals)) if pvals[x][1]>0.01]
 print "likelihood optimal k"
 print min(mins)+2
 print "optimal k"
 print bic.index(min(bic))+2
 gmm= mixture.GMM(n_components=bic.index(min(bic))+2,n_iter=100000,n_init=10, covariance_type='full')
 print gmm.get_params()
 gmm.fit(np.array(d))
 labs=gmm.predict(np.array(d))
 scores=gmm.score(np.array(d))

 if plot==True:
  fig,axes = plt.subplots(nrows=bic.index(min(bic))+2,ncols=1)
  for i in range(len(axes)):
       syl = []
       for x in np.where(labs==i)[0]: syl.extend(np.pad(array_of_syls[x],(160,160),'constant',constant_values=(0,0)))
       axes[i].specgram(syl,NFFT=512,Fs=32000,noverlap=510)
  plt.tight_layout()

 return segedpsds,labs,scores,bic.index(min(bic))+2,min(mins)+2

def eucliddist(a,b):
 """calculates a euclidean distance between two equal length arrays"""
 return np.sqrt(sum((np.array(a)-np.array(b))**2))

def minkowskidist(a,b,p):
 """calculates a minkowski distance between two equal length arrays"""
 return sum((np.array(a)-np.array(b))**p)**(1/float(p))

def sqr_eucliddist(a,b):
 """calculates a euclidean distance between two equal length arrays"""
 return (sum((np.array(a)-np.array(b))**2))

def mahalanobisdist(a,b):
 """calculates the mahalanobis distance between tow equal lenght vectors"""
 a=np.array(a)
 b=np.array(b)
 s=np.cov(np.array([a,b]).T)
 try: sinv=np.linalg.inv(s)
 except  np.linalg.LinAlgError: sinv=np.linalg.pinv(s)
 return sc.spatial.distance.mahalanobis(a,b,sinv)

def mahaldist(a,b):
 a=np.array(a)
 b=np.array(b)
 s=np.cov(np.array([a,b]).T)
 xminy=np.array(a-b)
 try: sinv=np.linalg.inv(s)
 except  np.linalg.LinAlgError: sinv=np.linalg.pinv(s)
 m=abs(np.dot(np.dot(xminy,sinv),xminy))
 return np.sqrt(m)

def pearsonrcoeff(a,b):
 return pearsonr(a,b)[0]

def spearmanrcoeff(a,b):
 return spearmanr(a,b)[0]

def sqformdistmat(array):
 """creates a squareform distance matrix for an array.  Currently only
 uses euclidean dist"""
 out=[]
 for x in array:
  intout=[]
  for y in array:
    intout.append(sqr_eucliddist(x,y))
  out.append(intout)
 return out

def norm(a):
 """normalizes a string by it's average and sd"""
 a=(np.array(a)-np.average(a))/np.std(a)
 return a

def sigmoid_norm(x):
 """returns a sigmoid nomalization by the sigmoid function"""
 return 1 / (1 + np.exp(-np.asarray(x)))

def main():
 path='/opt/data/pk31gr76/saline_8_9_2016/'
 batchfile = path+'batch.keep'
 ff = open(batchfile)
 files = [line.rstrip() for line in ff]
 ff.close()
 from multiprocessing import Pool
 import analyze_songs
 pool = Pool(processes = int(1))
 results = []
 for ksong,song_file in enumerate(files):
  results.append(pool.apply_async(analyze_songs.analyze_song, (path+song_file,), dict(use_evsonganaly = True, filetype = 'cbin')))
 sylables = []
 sylable_ids = []
 song_data_rec = []
 for ksong, result in enumerate(results):
  song, syls, song_data = result.get()
  sylables.extend(syls)
  sylable_ids.extend([ksong]*len(syls))
  song_data_rec.append(song_data)
 all_labels = np.unicode(u'')
 for song_data in song_data_rec:
     all_labels=all_labels+song_data['labels'][0]
 syll_types = [i for i in set(all_labels) if i.isalpha()]
 segedpsds,labs,scores,bic,likli = EMofgmmcluster(sylables,syll_types,all_labels,plot=False)
 return segedpsds, labs, scores, bic, likli, sylables, syll_types

#main program
if __name__ =='__main__':
 main()
