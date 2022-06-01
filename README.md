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




