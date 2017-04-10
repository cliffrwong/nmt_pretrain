# NMT_Pretraining
Pretrain LSTM and embedding weights for classification of multilingual text.

This is a NMT except without attention. And the weights for the encoder and decoder's LSTM and embedding can be shared with the --shared flag. The last state of encoder is the input state of the decoder. The model is currently a many-to-one model where the encoder takes in different languages as input and they all try to predict the English translation in the decoder's output vectors. This model is similar to (Google's Multilingual NMT system)[https://arxiv.org/abs/1611.04558] except without attention



To train:

The data directory should contain pairs of language  files (the foreign language and English) that share the same file prefix but have the following suffix patterns:
```
fileprefix.<lang1>.<lang2>
```
Where lang1 is the code for the language pair that this file belongs to and lang2 is the actual language of the file. Every language pair will have 'en' for one of the languages pairs because we are "translating" every language into English.

```
python nmt_pretrain.py --data_dir <data directory containing pairs of language files> --train_dir <training/log dir>

```


To export trained weights as numpy arrays:
```
python nmt_pretrain.py --export --data_dir <data directory containing quality estimation data> --train_dir <training dir>

```

