# PyTorch Dataloader for Slakh2100

Slakh2100 has a very challenging [file structure](https://github.com/ethman/slakh-utils#at-a-glance). Each track consists of different number of sources. i.e. different number of midi and flac tracks for each `mix.flac`:
```
Track00001
   └─── all_src.mid
   └─── metadata.yaml
   └─── MIDI
   │    └─── S01.mid
   │    │    ...
   │    └─── SXX.mid
   └─── mix.flac
   └─── stems
        └─── S01.flac
        │    ...
        └─── SXX.flac 
```

This dataset is suitable for the following tasks:
- Music Instrument Recognition
- Automatic Music Transcription (AMT)
- Music Source Separation (MSS)

## Preprocessing
### From redux to original
To convert Slakh2100 back to the original version, you can run this command:
```bash
python resplit_slakh.py -d /path/to/slakh2100/ -r
```

For more information about different versions of Slakh2100, please refer to this [page](https://github.com/ethman/slakh-utils#make-splits).

### Audio and midi packing
Assume that you have the dataset folder inside this repo, if you want to make use of all your CPU threads (`num_workers = -1`) preprocess the audio files into `16kHz` sampling rate and save them into a folder called `waveforms`:

```python
from slakh_loader.preprocessing import pack_audio_clips_multithread 
pack_audio_clips_multithread(input_dir = './slakh2100_flac_redux/',
                 output_dir = 'waveforms',
                 sample_rate = 16000,
                 num_workers = -1
                )
```


If you want to preprocess the train set of midi files:
```python
from slakh_loader.preprocessing import create_notes
create_notes('./slakh2100_flac_redux/', 
             './', 
             'train')
```

By default, each `SXX.flac` and `SXX.mid` corresponds to one Komplete 12 plugin/patch defined in this [json file](https://github.com/ethman/slakh-generation/blob/e6454eb57a3683b99cdd16695fe652f83b75bb14/instr_defs_metadata/komplete_strict.json). We map these plugins/patches into MIDI instruments based on our custom [MIDI map](./slakh_loader/MIDI_program_map.tsv). Based on this custom map, all `SXX.mid` for each track become one `TrackXXXXX.pkl` file under the `instruments_classification_notes_MIDI_class` folder; all `SXX.flac` for each track are remapped to:

```
waveforms
   └─── train
   │    └─── Track00001
   │         └─── Bass.flac
   │         └─── Drums.flac
   │         │    ...
   │         └─── Voiceflac 
   │         └─── waveform.flac   
   │
   └─── validation
   └─── test
```

The remapping of `SXX.flac` also includes audio downsampling to 16kHz. So we also have `waveform.flac` as the downsampled version of `mix.flac`



## Loading method
Each `idx` under the `__getitem__` function corresponds to each track `TrackXXXXX` in the dataset. Then, the audio mix `waveform.flac` is loaded as the `waveform` key.

The dataset can be loaded using:

```python
from slakh_loader.slakh2100 import Slakh2100
from slakh_loader.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
                                      
dataset = Slakh2100(slakhdata_root='./slakh2100_flac_redux',
          download= False,
          split= 'train',
          waveform_dir='./packed_waveforms',
          pkl_dir='./packed_pkl',
          segment_seconds=11,
          frames_per_second = 100,
          transcription = True,
          random_crop = False,
          source = True ,
          name_to_ix=MIDIClassName2class_idx,
          ix_to_name=class_idx2MIDIClass,
          plugin_labels_num=MIDI_Class_NUM-1,
          sample_rate=16000,             
          )                                   
                                      
```

`dataset[0]` returns a dictionary consisting of the following keys: `target_dict`, `sources`, `source_masks`, `waveform`, `start_sample`, `valid_length`, `flac_name`, `instruments`, `plugin_id`.

### Data Keys

[target_dict](./examples/target_dict.md) contains the frame rolls, onset rolls, and masks for different sources, and have the following hierarchy:
```
target_dict
   └─── Electric Guitar
   │    └─── onset_roll
   │    └─── reg_onset_roll
   │    └─── frame_roll
   │    └─── mask_roll 
   └─── Organ
   │    └─── ...   
   │    ...
   └─── Voice
```

[sources](./examples/sources.md) contains the waveforms for each source (loaded from the `./waveforms` folder).

[source_masks](./examples/source_masks.md) contains the boolean value for each source indicating if the waveform is available or not. Because there are more MIDI data than the audio data, there are MIDI tracks without a corresponding audio. [More information](https://github.com/ethman/slakh-utils/issues/20).

[waveform](./examples/waveform.md) contains the downsampled version of audio mix `waveform.flac`. It should be the model input for either __Music Instrument Recognition__, __Automatic Music Transcription (AMT)__, or __Music Source Separation (MSS)__.

[start_sample](./examples/start_sample.md) is an `int` which indicates the starting sample for the loaded audio segment. This key is only valid when `segment_seconds!=None`, where `Slakh2100` will randomly sample a segment from each `waveform.flac` inside `waveforms/train/TrackXXXXX` starting from `start_sample` with a duration `valid_length = sample_rate * segment_seconds`.

[valid_length](./examples/valid_length.md) indicates the duration of the segment. It is calcuated by `valid_length = sample_rate * segment_seconds`.

[flac_name](./examples/flac_name.md) indicates which track is being located.

[instruments](./examples/instruments.md) indicates which musical instruments are in the loaded segment. The dimension of it is determined by `plugin_labels_num`, `name_to_ix`, and `ix_to_name`. Please refer to [this section](#Arguments) for more information about these arguments.

[plugin_id](./examples/plugin_id.md) has no practical use in this current release. It is a placeholder for future expansion. 

### Arguments

`waveform_dir`: The location of the `waveform` folder mentioned in the [preprocessing](#Preprocessing) section.

`notes_pkls_dir`: The location of the `instruments_classification_notes_MIDI_class` folder mentioned in the [preprocessing](#Preprocessing) section.

`segment_seconds`: The segment length of the audio clip sampled from the full audio. If `segment_seconds=None`, then a full audio is loaded instead.

`frames_per_second`: Number of spectrogram frames per second. It is for extracting a correct piano roll matching with the spectrogram frames. If `sample_rate=16000` and `hop_size=160`, then `frames_per_second=16000/160` which is `100`.

`transcription`: A boolean to determine whether to load piano rolls for transcription task.

`random_crop`: A boolean to determine if an audio clips should be randomly sampled from the full audio. If it is set to `False`, it will always load the first `segment_seconds` of audio from the full audio track.

`source`: A boolean value to determine whether to load the sources of the audio tracks for music source separation.

`name_to_ix`: A dictionary that maps instrument names to theirs indices. The dictionary can be obtained by `from slakh_loader.MIDI_program_map import MIDIClassName2class_idx` which has an extra `empty` key for future expansion.

`ix_to_name`: A dictionary that maps instrument indices to their names. . The dictionary can be obtained by `from slakh_loader.MIDI_program_map import class_idx2MIDIClass` which has an extra `empty` key for future expansion.

`plugin_labels_num`: The number valid instrument labels for `name_to_ix` and `ix_to_name`. If `plugin_labels_num < len(name_to_ix)` or `plugin_labels_num < len(ix_to_name)`, then only the first `plugin_labels_num` keys in `name_to_ix` and `ix_to_name` will be used.

`sample_rate`: The sampling rate for the audio clips.

## Packing data into a batch v1
Assume that you have successfullly created the `dataset` object as described [here](#Loading-method). After that, you need to use `End2EndBatchDataPreprocessor` to help packing the dictionary into a batch.

`End2EndBatchDataPreprocessor` randomly samples stems (dictionary key `sources`) and piano rolls (dictionary key `target_dict`) from each track based on the `samples` parameter.

```
from slakh_loader.slakh2100 import collate_slakh, End2EndBatchDataPreprocessor
batch_processor = End2EndBatchDataPreprocessor(name_to_ix=MIDIClassName2class_idx,
                                               ix_to_name=class_idx2MIDIClass,
                                               plugin_labels_num=MIDI_Class_NUM-1,
                                               mode='imbalance',
                                               temp=0.5,
                                               samples=3,
                                               neg_samples=1,
                                               audio_noise=0,
                                               transcription=True,
                                               source_separation=False)


loader = DataLoader(dataset, batch_size=2, collate_fn=collate_slakh)
batch_dict = next(iter(loader))

                                               
batch = batch_processor(batch_dict)    
```

## Packing data into a batch v2
```
from slakh_loader.slakh2100 import SlakhCollator


loader = DataLoader(dataset, batch_size=2, collate_fn=collate_slakh)
batch_dict = next(iter(loader))

                                               
batch = batch_processor(batch_dict)    
```

