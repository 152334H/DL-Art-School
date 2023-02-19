#!/usr/bin/env sh

#source this in your profile

TORTOISE_DIR="/home/$USER/code/tortoise-tts-fast/"
TORTRAIN_DIR="/home/$USER/code/DL-Art-School/"
TTS_TEXT="or a given problem should be approached using traditional or conventional machine learning or should I be doing deep learning."
EXPERIMENT_NAME="dade"

tortoise_get_voice_pth_as(){
     cd $($TORTOISE_DIR)tortoise
     python get_conditioning_latents.py --voice $1
     cd $($TORTOISE_DIR)tortoise/voices/
     mkdir $2
     cd $2
     cp $($TORTOISE_DIR)results/conditioning_latents/$1.pth ./$2.pth
}

alias example="--voice emma --seed 42 --text '$TTS_TEXT'"
alias tortoise="cd $TORTOISE_DIR'' && python tortoise/do_tts.py --kv_cache --half --preset very_fast"

tortoise_use_train(){
     alias tortoise_use="tortoise --text '$TTS_TEXT' --ar-checkpoint"
     tortoise_use $(tortoise_get_latest_train) $1
}

tortoise_train(){
     cd $TORTRAIN_DIR"codes"
     python3 train.py -opt ../experiments/EXAMPLE_gpt.yml
}

tortoise_quick_experiment(){
     bash $TORTRAIN_DIR"quick-experiment.sh"
}

tortoise_get_latest_train(){
     models_place=$TORTRAIN_DIR"experiments/$EXPERIMENT_NAME/models/"
     cd $models_place
     latest_model=$(ls | tail -1)
     echo $models_place$latest_model
}
