import math
from pprint import pprint
from typing import List, Tuple, Any, Union, Iterable
from EntropyHub import SampEn
import PyEMD
import mne
import numpy as np
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
from scipy.stats import kurtosis
from mne.io import RawArray, Raw
from mne.preprocessing import ICA
from sklearn import preprocessing

freq = 250


def one_d_emd_decompose(data):
    eemd = PyEMD.EEMD(trials=15)
    eemd.eemd(data)
    imfs, res = eemd.get_imfs_and_residue()
    return imfs, res


def get_imfs(data, ch_names):
    channel_number, data_length = data.shape
    pprint("artifact_removal, channel_number: " + str(channel_number) + " data_length: " + str(data_length))
    imf_data = None
    imf_ch_names = []
    residual = []
    for i in range(0, channel_number):
        channel_data = data[i]
        imfs, res = one_d_emd_decompose(channel_data)
        residual.append(res)
        imf_number = imfs.shape[0]
        ch_name = ch_names[i]
        ch_name = ch_name.replace("EEG", "").strip()
        for i in range(0, imf_number):
            imf_ch_names.append(ch_name + " imf" + str(i + 1))
        if imf_data is None:
            imf_data = imfs
        else:
            imf_data = np.vstack((imf_data, imfs))
    return imf_data, imf_ch_names, residual


def one_d_imf_sqa(imf) -> Tuple[Any, Union[ndarray, Iterable, int, float]]:
    data = np.squeeze(imf)
    tolerance = np.std(data) * 0.15
    (res, *_) = SampEn(data, m=3, r=tolerance)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    return entropy, kurt


def ica_component_sqa(ica_component) -> Tuple[float, float]:
    data = np.squeeze(ica_component)
    tolerance = np.std(data) * 0.15
    (res, *_) = SampEn(data, m=3, r=tolerance)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    return entropy, kurt


def imf_arranging(data, ch_names) -> Tuple[
    List[int], List[int]]:
    res = DataFrame(data={'ch_name': [], 'data': []})
    channel_number = data.shape[0]
    for i in range(0, channel_number):
        res = pd.concat([
            DataFrame(data={
                'ch_name': [ch_names[i]],
                'data': [data[i]],

            }), res], sort=False, ignore_index=True
        )
    pprint(res)
    return


def imf_filtering(data, ch_names, entropy_threshold=0.5, kurt_threshold=1.5, no_filtering=False, logging=True) -> Tuple[
    List[
        int], List[int]]:
    print('entropy_threshold ' + str(entropy_threshold), 'kurt_threshold ' + str(kurt_threshold))
    res = DataFrame(data={'ch_name': [], 'entropy': [], 'kurt': []})
    channel_number = data.shape[0]
    for i in range(0, channel_number):
        entropy, kurt = one_d_imf_sqa(data[i])
        res = pd.concat([
            DataFrame(data={
                'ch_name': [ch_names[i]],
                'entropy': [entropy],
                'kurt': [kurt]
            }), res], sort=False, ignore_index=True
        )
    res['entropy'] = preprocessing.scale(res['entropy'])
    res['kurt'] = preprocessing.scale(res['kurt'])

    # filtered_imf_info = pd.concat([
    #     res.query(
    #         'entropy < {} and kurt > {} and kurt < {}'.format(-entropy_threshold, -1.5, 1.5)
    #     ),
    #     res.query(
    #         'entropy > {} and kurt > {} and kurt < {}'.format(entropy_threshold, -1.5, 1.5)
    #     )
    # ])
    filtered_imf_info = res.query(
        'entropy > {} and entropy < {} and kurt > {} and kurt < {}'
            .format(-entropy_threshold, entropy_threshold, -kurt_threshold, kurt_threshold)
    )
    pprint(res)
    return list(filtered_imf_info.index), list(set(res.index).difference(set(filtered_imf_info.index)))


def ICA_decompose(data, input_ch_names, data_freq=freq) -> Tuple[Any, ICA, RawArray]:
    ch_names = input_ch_names if not isinstance(input_ch_names, np.ndarray) else list(input_ch_names)
    new_info = mne.create_info(ch_names, ch_types=["eeg"] * len(ch_names), sfreq=data_freq)
    raw = mne.io.RawArray(data, new_info)
    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    n_components = 14
    # n_components = int(len(input_ch_names) * 0.9)
    ica = ICA(n_components=n_components, max_iter='auto', random_state=97, method='picard')
    ica.fit(filt_raw)
    ica_components = ica.get_sources(raw)
    return ica_components, ica, raw


def ica_stage(data, input_ch_names) -> Raw:
    ica_components_raw, ica, data_raw = ICA_decompose(data, input_ch_names)
    print('ica_components_raw: ', ica_components_raw.copy())
    segemntment_len = int(freq * 0.5)
    picked_freq = np.zeros(len(input_ch_names))
    total_freq = math.ceil(len(ica_components_raw.get_data()[0]) / segemntment_len)
    for start_index in range(0, len(ica_components_raw.get_data()[0]), segemntment_len):
        tmin = start_index
        tmax = (start_index + segemntment_len)
        new_info = mne.create_info(ica_components_raw.ch_names, ch_types=["eeg"] * len(ica_components_raw.ch_names),
                                   sfreq=freq)
        raw_slice = mne.io.RawArray(ica_components_raw.get_data()[:, tmin: tmax], new_info)
        picked_indexes = np.array(ica_components_filtering(
            raw_slice,
            input_ch_names, entropy_threshold=1.5, kurt_threshold=1.5
        ))
        picked_freq[picked_indexes] += 1
    picked_freq = picked_freq / total_freq
    picked_indexes = np.where(picked_freq >= 0.8)
    print('picked_freq: ', picked_freq)
    print('ICA filtering: ', len(picked_indexes), '/', len(ica.ch_names))
    filtered_imfs = ica.apply(data_raw, include=picked_indexes)
    return filtered_imfs


def ica_components_filtering(ica_components_raw, ch_names, entropy_threshold=1.5, kurt_threshold=1.5,
                             no_filtering=False,
                             logging=True):
    data = ica_components_raw.get_data()
    res = DataFrame(data={'ch_name': [], 'entropy': [], 'kurt': []})
    channel_number = data.shape[0]
    for i in range(0, channel_number):
        entropy, kurt = ica_component_sqa(data[i])
        res = pd.concat([
            DataFrame(data={
                'ch_name': ['component ' + str(i + 1)],
                'entropy': [entropy],
                'kurt': [kurt]
            }), res], sort=False, ignore_index=True
        )
    res['entropy'] = preprocessing.scale(res['entropy'])
    res['kurt'] = preprocessing.scale(res['kurt'])
    # filtered_ica_component_info = pd.concat([
    #     res.query(
    #         'entropy < {} and kurt > {} and kurt < {}'.format(-entropy_threshold, -kurt_threshold, kurt_threshold)
    #     ),
    #     res.query(
    #         'entropy > {} and kurt > {} and kurt < {}'.format(entropy_threshold, -kurt_threshold, kurt_threshold)
    #     )
    # ])
    filtered_ica_component_info = res.query(
        'entropy > {} and entropy < {} and kurt > {} and kurt < {}'
            .format(-entropy_threshold, entropy_threshold, -kurt_threshold, kurt_threshold)
    )
    print('IC sqa:')
    pprint(res)
    # filtered_ica_component_info = res.query('entropy < {} and kurt < {}'.format(entropy_threshold, kurt_threshold))
    return filtered_ica_component_info.index


def imfs_merge(imf_raw: Raw):
    imfs = imf_raw.get_data()
    imf_names = imf_raw.info['ch_names']
    df = DataFrame(columns=['imf_name'])
    for i in range(0, len(imf_names)):
        imf_name = imf_names[i].split(' ')[0]
        df = pd.concat((
            df,
            DataFrame(
                data=[{
                    'imf_name': imf_name
                }]
            )
        ), ignore_index=True)
    group_by = df.groupby('imf_name')
    groups = group_by.groups
    eeg_channels = None
    for key in groups.keys():
        indexes = groups[key]
        imfs_of_this_ch = imfs[indexes]
        eeg_channel = np.zeros(shape=(1, imfs.shape[1]))
        for i in range(0, len(indexes)):
            eeg_channel += np.array(imfs_of_this_ch[i])
        if eeg_channels is None:
            eeg_channels = eeg_channel
        else:
            eeg_channels = np.vstack((eeg_channels, eeg_channel))
    ch_names = list(groups.keys())
    new_info = mne.create_info(ch_names, ch_types=["eeg"] * len(ch_names), sfreq=freq)
    raw = mne.io.RawArray(eeg_channels, new_info)
    print(eeg_channels.shape)
    return raw
