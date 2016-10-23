import numpy as np
import scipy as sp
from multiprocessing import Pool

import songtools2 as songtools
import sylcluster_gaussian_mixture_expectation_max_dbscan_kfoldcv_noplot_50x as cluster
import evsonganaly


def analyze_song(song_file, use_evsonganaly = False, filetype = 'wav'):
    if filetype=='wav':
        song = songtools.impwav(song_file)
    elif filetype=='cbin':
        song = songtools.impcbin(song_file)
    elif filetype=='raw':
        song = songtools.impraw(song_file)
    elif filetype == 'int16':
        song = songtools.impmouseraw(song_file)
    else:
        raise Exception('Filetype ' + filetype + ' not supported')
    if use_evsonganaly:
        song_data = evsonganaly.load_ev_file(song_file, load_song = False)
        song_data['a'] = song
        syls, labels = evsonganaly.get_ev_sylables(song_data)
    else:
        sm_win = 2
        min_dur = 0.01
        song_data = {}
        song_data['threshold']=.5 # threhold in units of std
        syls, times = songtools.getsyls(song, min_length = min_dur, window = sm_win, threshold = song_data['threshold'])
        song_data['Fs'] = float(song[1])
        song_data['fname'] = song_file
        song_data['onsets'] = np.array([syl[0] for syl in times])*1e3
        song_data['offsets'] = np.array([syl[1] for syl in times])*1e3
        song_data['sm_win'] = sm_win
        song_data['min_dur'] = min_dur * 1e3
        song_data['min_int'] = 0

    return song, syls, song_data

def analyze_and_label_songs(song_files, run_name = '', plot = False, n_processors = 1, use_evsonganaly = False, use_autodata_dir = True, xcorr_allign = False, filetype = 'wav', do_model_selection = False, n_models=10):
    # map songs to pool to gather song data
    pool = Pool(processes = int(n_processors))
    results = []
    for ksong, song_file in enumerate(song_files):
        results.append(pool.apply_async(analyze_song, (song_file,), dict(use_evsonganaly = use_evsonganaly, filetype = filetype)))
        # song, syls ,song_data = analyze_song(song_file, use_evsonganaly = use_evsonganaly, filetype = filetype) # this is here for testing without pool
        #import ipdb; ipdb.set_trace()

    # gather song data from results
    sylables = []
    sylable_ids = []
    song_data_rec = []
    song_files = song_files
    for ksong, result in enumerate(results):
        song, syls, song_data = result.get()
        sylables.extend(syls)
        sylable_ids.extend([ksong]*len(syls))
        song_data_rec.append(song_data)
    sylable_ids = np.array(sylable_ids)
    fs = song[1]
    if xcorr_allign:
        sylables = songtools.allign_by_xcorr(sylables)

    # cluster syllables
    model_selection_data, labels, PSDMAT, freq = cluster.EMofgmmcluster(sylables, n_processors = n_processors, fs = fs, do_model_selection = do_model_selection, n_models = n_models)
    # gather labels into song_data dicts
    list_of_song_labels = []
    for ksong, song_file in enumerate(song_files):
        song_labels = labels[sylable_ids==ksong]
        song_data_rec[ksong]['labels']=np.array(''.join(letter_labels[song_labels]))
        evsonganaly.save_ev_file(song_data_rec[ksong], use_autodata_dir = use_autodata_dir)
        list_of_song_labels.append(''.join(letter_labels[song_labels]))

    # split up the motifs
    motifs, motif_song_idxs = sequence.split_up_motifs_with_hmm(list_of_song_labels)
    motifs = np.array(motifs)
    motif_song_idxs = np.array(motif_song_idxs)
    # load motif data into song_data dicts
    for ksong in range(0,len(song_data_rec)):
        song_data_rec[ksong]['motifs'] = list(motifs[motif_song_idxs==ksong])

    # save song_data dicts to evfiles
    for ksong in range(0,len(song_data_rec)):
        evsonganaly.save_ev_file(song_data_rec[ksong], use_autodata_dir = use_autodata_dir)
    # save model_selection data
    path = song_data_rec[0]['fname']
    dirpath = path[:path.rfind('/')+1]
    if not os.path.exists(dirpath + 'autodata/'):
        os.mkdir(dirpath + 'autodata/')
    sp.io.savemat(dirpath+'autodata/'+run_name+'model_selection_run.mat', model_selection_data)
    return sylables, PSDMAT, labels

if __name__=="__main__":
    path='/opt/data/pk31gr76/saline_8_9_2016'
    batchfile = path+'/batch.keep'
    ff = open(batchfile)
    files = [line.rstrip() for line in ff]
    ff.close()
    filename = path.split('/')[-1:][0]
    sylpath = path+'/summarydat_30foldxv_50_syls/'
    if os.path.exists(sylpath)==False: os.mkdir(sylpath)
    sylables, psdmat, labels = analyze_and_label_songs(files,use_evsonganaly=True,filetype='cbin')