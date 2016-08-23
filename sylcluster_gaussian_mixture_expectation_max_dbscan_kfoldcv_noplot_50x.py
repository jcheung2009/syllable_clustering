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
import entropy_estimators as ee


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

def DBSCANcluster(array_of_syls):
 """takes an array of segmented sylables and clusters them by 
 taking psds (welch method) using DBSCAN"""
 nfft=30000
 fs=32000
 segstart=int(600/(fs/nfft))
 segend=int(16000/(fs/nfft))
 welchpsds=[psd(x,NFFT=nfft,Fs=fs) for x in array_of_syls]
 segedpsds=[x[0][segstart:segend] for x in welchpsds] #removes any values below 650 and above 10k
 d=sqformdistmat([norm(x) for x in segedpsds])
 #d=sqformdistmat([x for x in segedpsds])
 s=1-(d/np.max(d))
 db=skl.cluster.DBSCAN(eps=0.99, min_samples=5).fit(s) 
 labels = db.labels_
 # Number of clusters in labels, ignoring noise if present.
 n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
 no_syls=len(array_of_syls)
 no_clustered_syls=len([x for x in labels if -1 != x])
 percent_clustered=(float(no_clustered_syls)/float(no_syls))
 print min(labels)
 print max(labels)
 print 'Number of sylables %d' % no_syls
 print 'Number of clustered sylables %d' % no_clustered_syls
 print 'Percent of sylables clustered %0.3f' % percent_clustered
 print 'Estimated number of clusters: %d' % n_clusters_
 print 'mean cluster size %0.3f' % np.average([list(labels).count(y) for y in range(0,int(max(labels))+1)])
 print 'size of each cluster'
 print [list(labels).count(y) for y in range(0,int(max(labels))+1)]
 return labels,segedpsds,s
 #print ("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(d, labels, metric='precomputed'))

def EMofgmmcluster(array_of_syls):
 """takes an array of segmented sylables and clusters them by 
 taking psds (welch method) and fitting a mixture model"""
 nfft=2**14
 fs=32000
 segstart=int(600/(fs/nfft))
 segend=int(16000/(fs/nfft))
 welchpsds=[psd(x,NFFT=nfft,Fs=fs) for x in array_of_syls]
 segedpsds=[norm(x[0][segstart:segend]) for x in welchpsds]
 models=range(2,16)
 #pca=skl.decomposition.PCA(n_components=16000)
 #segedpsds=pca.fit_transform(segedpsds)
 tim=float(tm.time())
 print np.shape(segedpsds[0])
 print len(segedpsds)
 d=sc.spatial.distance.squareform(sc.spatial.distance.pdist(segedpsds,'sqeuclidean'))
 #d=np.dot(np.array(segedpsds),np.array(segedpsds).T)
 d=d.T[:50].T
 #d=sqformdistmat(segedpsds)
 #d=1-d/np.max(d)
 #print np.shape(list(d))
 #print ee.mi(segedpsds,d)
 #print ee.kldiv(segedpsds,d)
 #print ee.kldiv(segedpsds,segedpsds)
 #print ee.mi(segedpsds,segedpsds)
 #print skl.metrics.cluster.entropy(d)
 #print skl.metrics.cluster.entropy(segedpsds)
 s=d
 #s=segedpsds
 #pca=decomposition.PCA(n_components=20)
 #pca.fit(s)
 #t=pca.transform(s)
 '''plt.plot(pca.explained_variance_ratio_,'o')
 plt.show()'''
 #print ee.mi(t,d)
 #print ee.mi(segedpsds,t) 
 #print np.shape(t)



# print float(tm.time())-tim
 #s1-norm(d)
 fits=[]
 loglikelihood=[]
 liklis=[]
 bic=[]
 aic=[]
 k=3
 reps=1
# dpgmm=mixture.DPGMM(n_components=20, covariance_type='full', alpha=100000 , n_iter=10000)
# dpgmm.fit(s)
# print dpgmm.predict(s)
# print max(dpgmm.predict(s))
# print min(dpgmm.predict(s))
 for x in models:
  errs=[]
  bics=[]
  aics=[]
  for m in range(reps):
   #s=1-(d/np.max(d))
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
  #print (np.mean(errs))
 #plt.show()
 bic=[np.mean(x) for x in bic]
 #plt.plot(bic)
 #plt.plot([np.mean(x) for x in aic])
# plt.plot([np.mean(x) for x in liklis])
 #plt.show()
 #plt.show() 
 #plt.plot(-2*np.array(loglikelihood))
 #plt.show()
 #plt.plot(np.diff(loglikelihood))
 #plt.show()
 #plt.hist(liklis[:9],bins=50,histtype='step')
 #plt.show()
 pvals=[sc.stats.mannwhitneyu(liklis[x],liklis[x+1]) for x in range(len(liklis)-1)]
 #plt.plot([x[0] for x in pvals])
 #plt.show()
 #plt.plot([x[1] for x in pvals])
 #plt.show()
 mins=[x for x in range(len(pvals)) if pvals[x][1]>0.01]
 print "lihlihood optimal k"
 print min(mins)+2
 print "optimal k"
 print bic.index(min(bic))+2
 gmm= mixture.GMM(n_components=bic.index(min(bic))+2,n_iter=100000,n_init=10, covariance_type='full')
 #gmm= mixture.GMM(6,n_iter=100000, covariance_type='diag')
 print gmm.get_params()
 #s=1-(d/np.max(d))
 gmm.fit(np.array(s))
 labs=gmm.predict(np.array(s))
 scores=gmm.score(np.array(s))
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



#main program
path=sys.argv[1]
files=os.listdir(path)
filename=path.split('/')[-1:][0]
files=[file for file in files if '.wav' == file[-4:]]
sylpath=path+'summarydat_30foldxv_50_syls/'
if os.path.exists(sylpath)==False: os.mkdir(sylpath)
clustno_summary_out=open(sylpath+filename+'clustno_summary_out.txt','w')



sylables=[]

fil_objs={}
for file in files[:3000]:
 print file
 song=impwav(path+file)
 #fsong=filtersong(song)
 if len(song)<1: break
 syls,objs=(getsyls(song))
 fil_objs[file]=objs
 for x in syls: sylables.append(x)

print len(sylables)
#print [len(x) for x in sylables]
segedpsds,labs, scores,sylno_bic,sylno_likli=EMofgmmcluster(sylables[:3000])

#segedpsds,labs, scores=EMofgmmcluster(sylables)


pk.dump(segedpsds,open(sylpath+filename+'psds.pk','w'))
pk.dump(labs,open(sylpath+filename+'labs.pk','w'))
pk.dump(scores,open(sylpath+filename+'scores.pk','w'))

pk.dump(fil_objs,open(sylpath+filename+'seged_dict.pk','w'))


print max(labs)
print min(labs)
print len(scores)
print len(sylables)

'''avsyls=[]
sylscore=[]
for x in range(-1,int(max(labs))+1):
 tmpsyls=[]
 tmpscore=[]
 for n in range(0,len(sylables)):
  if labs[n]==x:
   tmpsyls.append(list(sylables[n]))
   tmpscore.append(scores[n])
 avsyls.append(list(tmpsyls))
 sylscore.append(tmpscore)'''

pk.dump(sylables[:3000],open(sylpath+filename+'syls.pk','w'))
#print [len(x) for x in sylscore]

avpsds=[]
for x in range(-1,int(max(labs))+1):
 tmpsyls=[]
 for n in range(0,len(segedpsds)):
  if labs[n]==x:
   tmpsyls.append(np.array(segedpsds[n]))
 avpsds.append(np.array(tmpsyls))

'''for x in range(1,len(sylscore)):
 mx=max(sylscore[x])
 index=sylscore[x].index(mx)
 bestsyl=np.array(avsyls[x][index])
 outwave(sylpath+'bestsyl'+str(x)+'.wav',(32000,bestsyl))'''

clustno_summary_out.write(str(sylno_bic)+'\n')
clustno_summary_out.write(str(sylno_likli)+'\n')
clustno_summary_out.write(str(len(sylables))+'\n')
clustno_summary_out.write('50\n')
clustno_summary_out.close()


'''avpsds=[sum(np.array(x))/len(np.array(x)) for x in avpsds[1:]]
plt.show()

for x in avpsds: 
 plt.plot(x)
plt.show()

d=sqformdistmat([norm(x) for x in avpsds])
s=1-(d/np.max(d))
db=skl.cluster.DBSCAN(eps=0.99, min_samples=1).fit(s)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
no_syls=len(avpsds)
no_clustered_syls=len([x for x in labels if -1 != x])
percent_clustered=(float(no_clustered_syls)/float(no_syls))
print min(labels)
print max(labels)
print 'Number of sylables %d' % no_syls
print 'Number of clustered sylables %d' % no_clustered_syls
print 'Percent of sylables clustered %0.3f' % percent_clustered
print 'Estimated number of clusters: %d' % n_clusters_
print 'mean cluster size %0.3f' % np.average([list(labels).count(y) for y in range(0,int(max(labels))+1)])
print 'size of each cluster'
print [list(labels).count(y) for y in range(0,int(max(labels))+1)]


clusterassns=[]
for n in range(len(avpsds)):
 tmpassns=[]
 for m in range(int(max(labels)+1)):
  if labels[n]==m:
   clusterassns.append(m)

print 'cluster assignments'
print clusterassns'''


'''pltsyls=[]
print len(sylables)
for x in range(len(avsyls)):
 print len(avsyls[x])

for x in range(len(avsyls)):
 if len(avsyls[x])>300: n=300
 else: n=len(avsyls[x])
 for m in range(n):
  pltsyls=pltsyls+avsyls[x][m]
  pltsyls=pltsyls+list(np.zeros(100))
 pltsyls=pltsyls+list(np.zeros(500))
 #pltsyls=pltsyls+avsyls[x][5]
 #pltsyls=pltsyls+list(np.zeros(500))


plt.specgram(pltsyls,Fs=32000)

plt.show()'''


