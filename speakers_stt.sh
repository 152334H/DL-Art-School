#!/usr/bin/env sh

# Transcribe what you're hearing, one keybinding away
# Put something like this in your sway/config:
# bindsym $mod+Shift+return exec alacritty -e bash /home/tekakutli/code/DL-Art-School/speakers_stt.sh





#THIS ASUMES YOU ALREADY HAVE WHISPER.CPP INSTALLED, SOMEWHERE
#https://github.com/ggerganov/whisper.cpp
#SET THESE WITH YOURS
LANG_FROM="en"
PATH_TO_WHISPER="/home/$USER/code/whisper.cpp"
PATH_TO_MODELS="/home/$USER/files/models"
OFFSET=0
PATH_TRAINING="/tmp/val/"


##########################


CURRENTDIR=$(pwd)
mkdir -p $PATH_TRAINING
cd $PATH_TRAINING


METAFILE="./metadata.csv"
if [ ! -f $METAFILE ]
then
    touch $METAFILE
fi




STT_COUNTER=$(wc -l < metadata.csv)
echo $STT_COUNTER
STT_COUNTER=$(($STT_COUNTER+$OFFSET))


LIVE_RECORD="./recording.wav"
WAV_RECORD="./recording_whisper.wav"
pw-record $LIVE_RECORD
ffmpeg -y -i "$LIVE_RECORD" -ar 16000 -ac 1 -c:a pcm_s16le $WAV_RECORD



# TINY IS FAST AND ENOUGH
if [[ "$LANG_FROM" == "en" ]]; then
    $PATH_TO_WHISPER/main -m $PATH_TO_MODELS/ggml-tiny.bin -l en -f "$WAV_RECORD" -otxt
else
    $PATH_TO_WHISPER/main -m $PATH_TO_MODELS/ggml-large.bin -l $LANG_FROM -tr -f "$WAV_RECORD" -otxt
fi

mkdir -p "./wavs"

TTT_COUNTER="TTS-"$STT_COUNTER
# cp "$LIVE_RECORD" "./wavs/$TTT_COUNTER.wav"
cp "$LIVE_RECORD" "$TTT_COUNTER.wav"
echo "$TTT_COUNTER""|"$(cat "$WAV_RECORD".txt) >> metadata.csv

cd $CURRENTDIR
