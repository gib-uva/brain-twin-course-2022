# EXTERNAL MODULES
import numpy as np
from tabulate import tabulate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# MEDUSA MODULES
import medusa
from medusa import components
from medusa import meeg
from medusa.bci import erp_spellers
from medusa.plots import erp_plots
# OTHER
import utils

#%% LOAD DATA
"""Load data"""
rec = components.Recording.load('data/train.rcp.bson')

# Represent ERP
erp_plots.plot_erp(rec, 1)

# Recording exploration
n_trials = len(np.unique(rec.erpspellerdata.trial_idx))
n_stimuli = len(rec.erpspellerdata.onsets)
target_cmd = [rec.erpspellerdata.commands_info[0][str(c[0][1])]['label']
    for c in rec.erpspellerdata.spell_target]

# Print some info of the extracted features
data_exploration = [
    ['Trials', n_trials],
    ['Stimuli', n_stimuli],
    ['Target', ';'.join(target_cmd)],
]
print('\nData exploration:')
print(tabulate(data_exploration))

#%% PRE-PROCESSING

# Band pass filter
utils.plot_eeg(rec.eeg.signal, rec.eeg.fs, "Raw EEG")
bp_filt = medusa.IIRFilter(
    order=7,
    cutoff=[1, 10],
    btype='bandpass',
    axis=0
)
bp_filt.fit(rec.eeg.fs)
bp_filt.display()
rec.eeg.signal = bp_filt.transform(rec.eeg.signal)
utils.plot_eeg(rec.eeg.signal, rec.eeg.fs, "After IIR")

# Common average reference
rec.eeg.signal = medusa.car(rec.eeg.signal)

#%% FEATURE EXTRACTION

# Extract features
features = medusa.get_epochs_of_events(
    timestamps=rec.eeg.times,
    signal=rec.eeg.signal,
    onsets=rec.erpspellerdata.onsets,
    fs=rec.eeg.fs,
    w_epoch_t=[0, 1000],
    w_baseline_t=[-250, 0],
    norm='z')

# Resample each epoch to the target frequency
features = medusa.resample_epochs(
    features,
    t_window=[0, 1000],
    target_fs=20)

# Reshape epochs and concatenate the channels
features = np.squeeze(features.reshape((features.shape[0],
                                        features.shape[1] *
                                        features.shape[2], 1)))

#%% CLASSIFICATION

# Instantiate rLDA
classifier = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

# Fit classifier and save model
classifier.fit(features, rec.erpspellerdata.erp_labels)
utils.save_model('rlda', classifier)

# Training accuracy
y_pred = classifier.predict_proba(features)[:, 1]

# Command decoding
spell_result, spell_result_per_seq, __ = erp_spellers.decode_commands(
    scores=y_pred,
    paradigm_conf=[rec.erpspellerdata.paradigm_conf],
    run_idx=np.zeros_like(rec.erpspellerdata.trial_idx),
    trial_idx=rec.erpspellerdata.trial_idx,
    matrix_idx=rec.erpspellerdata.matrix_idx,
    level_idx=rec.erpspellerdata.level_idx,
    unit_idx=rec.erpspellerdata.unit_idx,
    sequence_idx=rec.erpspellerdata.sequence_idx,
    group_idx=rec.erpspellerdata.group_idx,
    batch_idx=rec.erpspellerdata.batch_idx
)

# Spell accuracy per number of sequences
spell_acc_per_seq = erp_spellers.command_decoding_accuracy_per_seq(
    spell_result_per_seq,
    [rec.erpspellerdata.spell_target]
)
utils.print_acc_per_seq(spell_acc_per_seq)

