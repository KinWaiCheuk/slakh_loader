from slakh_loader.preprocessing import create_notes_multithread, pack_audio_clips_multithread

# downsample and pack audio clips into 16000Hz
pack_audio_clips_multithread('./slakh2100_flac_redux/', './', 16000)

# pack midi files into pkl files
create_notes_multithread('./slakh2100_flac_redux/', './')