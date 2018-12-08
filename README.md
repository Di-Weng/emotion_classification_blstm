# emotion_classification_blstm

It is an implement of BLSTM model for speech emotion classification combined with website visualization.
## References
https://github.com/RayanWang/Speech_emotion_recognition_BLSTM

## To do
Train model with 5 emotions--strip out neutral.

## Usage
```python = 3.6.7``` ```keras=2.2.4``` ```tensorflow=1.8.0```
```
from predict import load_model,get_audioclass

Load model: 
model = load_model('model/best_model.h5')
get_result: 
predict_class,predict_prob = get_audioclass(model,wav_file_path):
get_allaudio:  
predict_class,predict_prob,result_dic = get_audioclass(model,wav_file_path,all = True):
result_dic: {class:prob}
```
