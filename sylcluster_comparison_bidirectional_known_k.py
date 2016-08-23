import sys as sys
import os as os
import scipy as sc
from scipy import io
from scipy.io import wavfile
from scipy import ndimage
from scipy import signal
from scipy import spatial
from pylab import specgram, psd
#from matplotlib.mlab import psd
import numpy as np
import sklearn as skl
from sklearn import cluster
from sklearn import metrics
from sklearn import hmm
from scipy import spatial
from scipy.stats.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import decomposition
import random as rnd
from array import array
import cPickle as pk
import marshal as ml
from mahotas import otsu


"""This script creates gaussian mixture models for two different songs then calculates the K-L divergence
between the two mixture models.  It expects you to know how many syllables there are in both songs.
It expects song data to be in WAV format.

usage: python sylcluster_comparison_bidirectional_known_k.py <folder of songs for bird1> <folder of songs for bird2> <number of syllables bird 1> <number of syllables bird 2>
example: python sylcluster_comparison_bidirectional_known_k.py bk1bk6/ bu32bk23/ 9 10

It returns (to standard output) a string in this format: filename1+'\t'+filename2+'\t'+str(len2)+'\t'+str(score1)+'\t'+str(score2)
Where filename1 is the name of the first bird, filename2 is the name of the second bird, len2 is the number of reference syllables used, score1 is the Dkl (P||Q),
score2 is Dkl (Q||P)"""


#functions

def impwav(a):
 """Imports a wave file as an array where a[1] 
 is the sampling frequency and a[0] is the data"""
 out=[]
 wav = sc.io.wavfile.read(a)
 out=[wav[1],wav[0]]
 return out

def norm(a):
 """normalizes a string by it's average and sd"""
 a=(np.array(a)-np.average(a))/np.std(a)
 return a

def filtersong(a):
 out=[]
 b=sc.signal.iirdesign(wp=0.04, ws=0.02, gpass=1, gstop=60,ftype='ellip')
 out.append(sc.signal.filtfilt(b[0],b[1],a[0]))
 out.append(a[1])
 return (out)

def threshold(a,thresh=None):
 """Returns a thresholded array of the same length as input
 with everything below a specific threshold set to 0.
 By default threshold is sigma."""
 if thresh==None: thresh = sc.std(a)
 out=np.where(abs(a)>thresh,a,np.zeros(a.shape))
 return out

def findobject(file):
 value=(otsu(np.array(file,dtype=np.uint64)))
 thresh=threshold(file,value/3)
 thresh=threshold(sc.ndimage.convolve(thresh,np.ones(512)),0.5)
 label=(sc.ndimage.label(thresh)[0])
 objs=sc.ndimage.find_objects(label)
 return(objs)

def smoothrect(a,window=None,freq=None):
 """smooths and rectifies a song.  Expects (data,samprate)"""
 if freq== None: freq=32000
 if window == None: window=2
 le=int(round(freq*window/1000))
 h=np.ones(le)/le
 smooth= np.convolve(h,abs(a))
 offset = int(round((len(smooth)-len(a))/2))
 smooth=smooth[(1+offset):(len(a)+offset)]
 return smooth

def getsyls(a):
 """takes a file read in with impwav and returns a list of sylables"""
 fa=filtersong(a)
 frq=a[1]
 a=a[0]
 frqs=frq/1000
 #objs=findobject(smoothrect(fa[0],20,frq)/np.median(smoothrect(fa[0],20,frq)))
 objs=findobject(smoothrect(fa[0],20,frq))
 sylables=[x for x in [a[y] for y in objs] if int(len(x))>(10*frqs)]
 objs=[y for y in objs if int(len(a[y]))>10*frqs]
 return sylables,objs,frq


#main program
path1=sys.argv[1]
path2=sys.argv[2]
fils1=[x for x in os.listdir(path1) if x[-4:]=='.wav' and 'hp' not in x]
fils2=[x for x in os.listdir(path2) if x[-4:]=='.wav' and 'hp' not in x]

filename1=path1.split('/')[-2]
filename2=path2.split('/')[-2]
#foldname=path2.split('/')[-3]
'''outpath=(path2+filename1+'_'+filename2+'comp_out/')
if not os.path.exists(outpath):
 os.mkdir(outpath)
outdict={}'''

k=int(sys.argv[3])
k2=int(sys.argv[4])

syls1=[]
objss1=[]
for file in fils1:
 song=impwav(path1+file)
 if len(song)<1: break
 syls,objs,frq=(getsyls(song))
 objss1.append(objs)
 syls1.append([frq]+syls)
 #for x in syls: syls1.append(x)

syls2=[]
objss2=[]
for file in fils2:
 song=impwav(path2+file)
 if len(song)<1: break
 syls,objs,frq=(getsyls(song))
 objss2.append(objs)
 syls2.append([frq]+syls)
 #for x in syls: syls2.append(x)
#print len(syls2)
#print len(syls1)
segedpsds1=[]
for x in syls1:
 fs=x[0]
 nfft=int(round(2**14/32000.0*fs))
 segstart=int(round(600/(fs/float(nfft))))
 segend=int(round(16000/(fs/float(nfft))))
 psds=[psd(norm(y),NFFT=nfft,Fs=fs) for y in x[1:]]
 spsds=[norm(n[0][segstart:segend]) for n in psds]
 for n in spsds: segedpsds1.append(n)

segedpsds2=[]
for x in syls2:
 fs=x[0]
 nfft=int(round(2**14/32000.0*fs))
 segstart=int(round(600/(fs/float(nfft))))
 segend=int(round(16000/(fs/float(nfft))))
 psds=[psd(norm(y),NFFT=nfft,Fs=fs) for y in x[1:]]
 spsds=[norm(n[0][segstart:segend]) for n in psds]
 for n in spsds: segedpsds2.append(n)


basis_set=segedpsds1[:50]

segedpsds_all=[]

for x in segedpsds1[:1000]:
 segedpsds_all.append(x)

for x in segedpsds2[:1000]:
 segedpsds_all.append(x)

d1=sc.spatial.distance.cdist(segedpsds1[:1000],basis_set,'sqeuclidean')


d1_2=sc.spatial.distance.cdist(segedpsds1[1000:2000],basis_set,'sqeuclidean')


d2=sc.spatial.distance.cdist(segedpsds2[:1000],basis_set,'sqeuclidean')

d2_2=sc.spatial.distance.cdist(segedpsds2[1000:2000],basis_set,'sqeuclidean')

mx=np.max([np.max(d1),np.max(d2),np.max(d1_2),np.max(d2_2)])

s1=1-(d1/mx)
s1_2=1-(d1_2/mx)
s2=1-(d2/mx)
s2_2=1-(d2_2/mx)

mod1=mixture.GMM(n_components=k,n_iter=100000,covariance_type='diag')
mod1.fit(s1)

mod2=mixture.GMM(n_components=k2,n_iter=100000,covariance_type='diag')
mod2.fit(s2[:1000])

len2=len(s2)
len1=len(d1)
#print len1

score1_1= mod1.score(s1_2)
score2_1=mod2.score(s1_2)

score1_2=mod1.score(s2_2)
score2_2=mod2.score(s2_2)

len2=float(len(basis_set))
len1=float(len(basis_set))



'''score1= np.log2(np.e)*(np.mean(score1_1)/len(s1_2_1))-(np.mean(score2_1)/len(s1_2_1))
score2= np.log2(np.e)*(np.mean(score2_2)/len(s2[len1:]))-(np.mean(score1_2)/len(s2[len1:]))
sem1=np.log2(np.e)*(sc.stats.sem(score1_1)/float(len(s1_2_1)))-(sc.stats.sem(score2_1)/float(len(s1_2_1)))
sem2=np.log2(np.e)*(sc.stats.sem(score2_2)/len(s2[len1:]))-(sc.stats.sem(score1_2)/len(s2[len1:]))'''

score1= np.log2(np.e)*((np.mean(score1_1))-(np.mean(score2_1)))
score2= np.log2(np.e)*((np.mean(score2_2))-(np.mean(score1_2)))
sem1=np.log2(np.e)*((sc.stats.sem(score1_1))-(sc.stats.sem(score2_1)))
sem2=np.log2(np.e)*((sc.stats.sem(score2_2))-(sc.stats.sem(score1_2)))

score1=score1/len1
score2=score2/len2
sem1=sem1/len1
sem2=sem2/len2


#lab= mod1.predict(s2)

print filename1+'\t'+filename2+'\t'+str(len2)+'\t'+str(score1)+'\t'+str(score2)+'\t'+str(sem1)+'\t'+str(sem2)

#print filename1+'\t'+filename2+'\t'+str(len2)+'\t'+str(score1)+'\t'+str(score2)+'\t'+str(sem1)+'\t'+str(sem2)+'\t'+str(np.mean(score1_1))+'\t'+str(np.mean(score2_1))+'\t'+str(np.mean(score1_2))+'\t'+str(np.mean(score2_2))

'''outdict['mod1']=mod1
outdict['mod2']=mod2
outdict['scores1']=score1_1
outdict['labels']=lab
outdict['modbd']=filename1
outdict['compbd']=filename2
pk.dump(outdict,open(outpath+'outdict','w'))'''


