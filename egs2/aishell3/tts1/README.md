# AISHELL3 RECIPE

This is the recipe of Mandrain multi-speaker TTS model with [aishell3](https://www.openslr.org/93/) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# INITIAL RESULTS

## Pretrained models

### aishell3_tts_train_raw_phn_pypinyin_g2p_phone_train.loss.best
- Tacotron2
- https://huggingface.co/ftshijt/ESPnet2_pretrained_model_ftshijt_aishell3_tts_train_raw_phn_pypinyin_g2p_phone_train.loss.best

./tts_wlsc.sh --inference_nj 1 --nj 1 --train_config conf/tuning/train_wlsc.yaml --teacher_dumpdir data --train_set train --valid-set test --test-sets test --g2p none --cleaner none --token_type phn --python python3.7 --inference_model latest.pth --use-sid true --stage 6 "$@"; exit $?

