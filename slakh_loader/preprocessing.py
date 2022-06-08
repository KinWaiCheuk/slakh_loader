import os
import torchaudio
import torchaudio.functional as F
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
from slakh_loader.MIDI_program_map import idx2instrument_class

import contextlib
import glob
import joblib
from joblib import Parallel, delayed
import numpy as np
import pathlib
import pickle
import pretty_midi
import time
from tqdm import tqdm
import yaml


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def pack_audio_clips(
    input_dir: str,
    output_dir: str,
    sample_rate: int,
    num_workers=-1
    ):
    """
    Pack and resample audio clips into sources
    input_dir: location of Slack2100 dataset
    output_dir: location of the output packed audio
    sample_rate: the sample rate of the output audio
    Returns:
        None
    """

    for split in ["train", "test", "validation"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        split_input_dir = os.path.join(input_dir, split)
        audio_names = sorted(os.listdir(split_input_dir))

#         print("------ Split: {} (Total: {} clips) ------".format(split, len(audio_names)))

        params = []
        for audio_name in audio_names:
            audio_path = os.path.join(split_input_dir, audio_name, "mix.flac")
            output_path = os.path.join(split_output_dir, audio_name)
            os.makedirs(output_path, exist_ok=True)

            param = (audio_path, output_path, audio_name, split, sample_rate)
            # E.g., (0, './datasets/dataset-slakh2100/slakh2100_flac/train/Track00001/mix.flac',
            # './workspaces/hdf5s/waveforms/train/Track00001.h5', 'Track00001', 'train', 16000)
            params.append(param)
        # Debug by uncomment the following code.
        # write_single_audio_to_hdf5(params[0])

        # Pack audio files to hdf5 files in parallel.
#         with ProcessPoolExecutor(max_workers=None) as pool:
#             pool.map(write_audio, params)
        with tqdm_joblib(tqdm(desc=f"Packing {split} set audio clips", total=len(params))) as progress_bar:
                Parallel(n_jobs=num_workers)\
                        (delayed(write_audio)(param) for param in params)     



def write_audio(param):
    r"""Write a single audio file into an hdf5 file.
    Args:
        param: (audio_index, audio_path, output_path, audio_name, split, sample_rate)
    Returns:
        None
    """
    
    [audio_path, output_path, audio_name, split, sample_rate] = param
    audio, sr = torchaudio.load(audio_path)
    audio = F.resample(audio.squeeze(0), sr, sample_rate)

    duration = len(audio) / sample_rate

    torchaudio.save(os.path.join(output_path, 'waveform.flac'),
                    audio.unsqueeze(0),
                    sample_rate)
    
    dirname = os.path.dirname(audio_path) # getting the folder for the audio
    with open(os.path.join(dirname, "metadata.yaml"), "r") as stream:
        stem_dict = yaml.safe_load(stream)['stems']

    source_tracks = {}        
    for source_key, item in stem_dict.items():
        if item['midi_saved'] and item['audio_rendered']: # When midi_save=False, there is no audio track
            source_name = idx2instrument_class[item['program_num']]
            audio, _ = torchaudio.load(os.path.join(dirname, 'stems', f"{source_key}.flac"))
            audio = F.resample(audio.squeeze(0), sr, sample_rate)
#             audio = audio.numpy()

            if source_name in source_tracks.keys():
                source_tracks[source_name] += audio
            else:              
                source_tracks[source_name] = audio
                
    for key, i in source_tracks.items():
        torchaudio.save(
            os.path.join(output_path, f'{key}.flac'),
            i.unsqueeze(0),
            sample_rate
        )
        
        
        
DRUMS_PLUGIN_NAMES = [
    'ar_modern_sparkle_kit_full',
    'ar_modern_white_kit_full',
    'funk_kit',
    'garage_kit_lite',
    'pop_kit',
    'session_kit_full',
    'stadium_kit_full',
    'street_knowledge_kit',
]


def create_notes(path_dataset,
                 workspace,
                 split):
    r"""Create list of notes information for instrument classification.
    Args:
        path_dataset: str, he path of the original dataset
        workspace: str
        split: str, 'train' | 'validation' | 'test'
    Returns:
        None
    """
    path_dataset = path_dataset
    workspace = workspace
    split = split

    # paths
    output_dir = os.path.join(workspace, 'pkl_MIDI_class', split)
    os.makedirs(output_dir, exist_ok=True)

    # MIDI file names.
    path_dataset_split = os.path.join(path_dataset, split)
    piecenames = os.listdir(path_dataset_split)
    piecenames = [x for x in piecenames if x[0] != "."]
    piecenames.sort()
    # print("total piece number in %s set: %d" % (split, len(piecenames))

    # output_list = []
    # instrument_set = set()

    for piecename in tqdm(piecenames):

        # Read metadata of an audio piece. The metadata includes plugin
        # names for all tracks.
        filename_info = os.path.join(path_dataset_split, piecename, "metadata.yaml")

        with open(filename_info, 'r') as stream:
            try:
                metadata = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Get the trackname.
        path_midi = os.path.join(path_dataset_split, piecename, "MIDI")
        tracknames = glob.glob(os.path.join(path_midi, "*.mid"))
        tracknames = [os.path.splitext(os.path.basename(x))[0] for x in tracknames]
        tracknames.sort()  # ["S01", "S02", "S03", ...]

        # placeholder for the processed MIDI
        filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", tracknames[0] + ".mid")

        note_event_list = []

        for trackname in tracknames:
            # E.g., "S00".

            plugin_name = metadata["stems"][trackname]["plugin_name"]
            program_num = metadata["stems"][trackname]["program_num"]
            

            plugin_name = os.path.splitext(os.path.basename(plugin_name))[0]
            # E.g., 'elektrik_guitar'.

            # Read MIDI file of a track
            filename_midi = os.path.join(path_dataset_split, piecename, "MIDI", trackname + ".mid")
            midi_data = pretty_midi.PrettyMIDI(filename_midi)

            if len(midi_data.instruments) > 1:
                raise Exception("multi-track midi")
                
            instr = midi_data.instruments[0]

            # Append all notes of a track to output_list if not drums.
            if plugin_name not in DRUMS_PLUGIN_NAMES:
                # instrument_set.add(program_num)

                for note in instr.notes:

                    # Lower an octave for bass.
                    if instr.program in range(32, 40):
                        pitch = note.pitch - 12
                    else:
                        pitch = note.pitch

                    note_event = {
                        'split': split,
                        'audio_name': piecename,
                        'plugin_name': idx2instrument_class[program_num],
                        'plugin_names': [idx2instrument_class[program_num]],
                        'start': note.start,
                        'end': note.end,
                        'pitch': pitch,
                        'velocity': note.velocity,
                    }

                    # Remove notes with MIDI pitches larger than 109 (very few).
                    if note.pitch < 109:
                        # output_list.append(note_event)
                        note_event_list.append(note_event)
            else:
                # instrument_set.add(program_num)
                for note in instr.notes:
                    # Parse only valid drum pitch. MIDI note 35-81
                    # But in FL studio, it supports 24-84?
                    # Better make the range a bit larger
                    if note.pitch>=24 and note.pitch<=90:
                        note_event = {
                            'split': split,
                            'audio_name': piecename,
                            'plugin_name': 'Drums',
                            'plugin_names': ['Drums'],
                            'start': note.start,
                            'end': note.end,
                            'pitch': note.pitch,
                            'velocity': note.velocity,
                        }

                    # Remove notes with MIDI pitches larger than 109 (very few).
                    if note.pitch < 109:
                        # output_list.append(note_event)
                        note_event_list.append(note_event)                
                

        
        note_event_list.sort(key=lambda note_event: note_event['start'])

        note_event_list = add2(note_event_list)
        # output_list += note_event_list

        # E.g., output_list looks like: [
        #     {'split': 'train', 'audio_name': 'Track00001', 'plugin_name':
        #      'elektrik_guitar', 'start': 0.7811, 'end': 1.2576, 'pitch': 64, 'velocity': 127,
        #     },
        #     ...
        #     {'split': 'train', 'audio_name': 'Track00003', 'plugin_name':
        #      'jazz_guitar2', 'start': 58.2242, 'end': 58.4500, 'pitch': 57, 'velocity': 100,
        #     },
        #     ...
        # ]

        output_path = os.path.join(output_dir, '{}.pkl'.format(pathlib.Path(piecename).stem))
        pickle.dump(note_event_list, open(output_path, 'wb'))
    # pickle.dump(instrument_set, open(f'{split}_set', 'wb'))



def add2(note_event_list):

    new_list = []

    for note_event in note_event_list:
        note_event['instruments_num'] = 1

    for i in range(1, len(note_event_list)):
        if note_event_list[i]['pitch'] == note_event_list[i - 1]['pitch']:
            if note_event_list[i]['start'] - note_event_list[i - 1]['start'] <= 0.05:
                new_plugin_names = note_event_list[i]['plugin_names'] + note_event_list[i - 1]['plugin_names']
                new_instruments_num = note_event_list[i - 1]['instruments_num'] + 1

                for j in range(note_event_list[i - 1]['instruments_num'] + 1):
                    note_event_list[i - j]['instruments_num'] = new_instruments_num
                    note_event_list[i - j]['plugin_names'] = new_plugin_names

    for note_event in note_event_list:
        if len(note_event['plugin_names']) > 1:
            plugin_names = list(set(note_event['plugin_names']))
            note_event['plugin_names'] = plugin_names
            note_event['instruments_num'] = len(plugin_names)

    # for i in range(1, len(note_event_list)):
    #     if len(note_event_list[i]['plugin_names']) == 3:
    #         from IPython import embed; embed(using=False); os._exit(0)

    # for i in range(5):
    #     cnt = 0
    #     for note_event in note_event_list:
    #         if note_event['instruments_num'] == i:
    #             cnt += 1
    #     print(cnt)
    
    return note_event_list        