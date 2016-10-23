import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

import songtools2 as songtools


def load_ev_file(wav_fname, load_song=False):
    if wav_fname[-8:] == '.not.mat':
        data = sp.io.loadmat(wav_fname)
    else:
        data = sp.io.loadmat(wav_fname + '.not.mat')

    data['fname'] = wav_fname
    if load_song:
        if 'wav' in wav_fname:
            a = songtools.impwav(wav_fname)
        elif 'cbin' in wav_fname:
            a = songtools.impcbin(wav_fname)
        data['a'] = songtools.filtersong(a)
        data['t'] = np.divide(np.arange(0, len(a[0])), np.float(a[1]))
    return data


def save_ev_file(song_data, use_autodata_dir=False):
    # check that song is not in .mat file
    if 'a' in song_data.keys():
        song_data.pop('a')
    if 't' in song_data.keys():
        song_data.pop('t')

    if use_autodata_dir:
        path = song_data['fname']
        dirpath = path[:path.rfind('/') + 1]
        fname = path[path.rfind('/'):]
        if not os.path.exists(dirpath + 'autodata/'):
            os.mkdir(dirpath + 'autodata/')
        sp.io.savemat(dirpath + 'autodata/' + fname + '.not.mat', song_data)
    else:
        sp.io.savemat(song_data['fname'] + '.not.mat', song_data)
    return None


def get_ev_sylables(ev_song_dict):
    # if 'a' not in ev_song_dict:
    #     if 'wav' in wav_fname:
    #         ev_song_dict['a'] = songtools.filtersong(songtools.impwav(wav_fname))
    #     elif 'cbin' in wav_fname:
    #         ev_song_dict['a'] = songtools.filtersong(songtools.impcbin(wav_fname))
    sylables = []
    labels = []
    for ksyl, onset in enumerate(ev_song_dict['onsets']):
        offset = ev_song_dict['offsets'][ksyl]
        onset_idx = round(ev_song_dict['Fs'] * float(onset) / 1000)
        offset_idx = round(ev_song_dict['Fs'] * float(offset) / 1000)
        sylables.append(ev_song_dict['a'][0][onset_idx:offset_idx])
        labels.append(ev_song_dict['labels'][0][ksyl])
    sylables = np.array(sylables)
    labels = np.array(labels)
    return sylables, labels


x = 1

if __name__ == "__main__":
    data = load_ev_file('/opt/data/pk31gr76/saline_8_9_2016/pk31gr76_090816_0801.4491.cbin',load_song=True)
    ev_syls, labels = get_ev_sylables(data)
    syls, syl_times = songtools.getsyls(data['a'])
    import ipdb;

    # ipdb.set_trace();

# for kt in np.arange(0,10,.5):
# 	fig = plt.figure
# 	plt.plot(data['t'], data['a'][0])
# 	for k,onset in enumerate(data['onsets']):
# 		offset = data['offsets'][k]
# 		plt.plot([float(onset)/1000, float(offset)/1000], [0, 0], 'r')
# 	for time in syl_times:
# 		plt.plot([time[0], time[1]], [0,0])

# 	plt.xlim([kt+0, kt+0.5])
# 	plt.show()
