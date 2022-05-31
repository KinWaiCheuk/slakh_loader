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

```
from slakh_loader.slakh2100 import Slakh2100
from slakh_loader.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )
                                      
dataset = Slakh2100('train',
          './waveforms/',
          './instruments_classification_notes_MIDI_class/',
          segment_seconds=11,
          frames_per_second = 100,
          transcription = True,
          random_crop = False,
          source = True ,
          name_to_ix=MIDIClassName2class_idx,
          ix_to_name=class_idx2MIDIClass,
          plugin_labels_num=MIDI_Class_NUM,
          sample_rate=16000,             
          )                                   
                                      
```

`dataset[0]` returns a dictionary consisting of the following keys: `target_dict`, `sources`, `source_masks`, `waveform`, `start_sample`, `valid_length`, `flac_name`, `instruments`, `plugin_id`.

[target_dict](./examples/target_dict.md): contains the frame rolls, onset rolls, and masks for different sources, and have the following .
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