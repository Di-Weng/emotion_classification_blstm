# emotion_classification_blstm

It is an implement of BLSTM model for speech emotion classification combined with website visualization.

## Dependencies

* python = 3.6.7
* Flask = 1.0.2
* keras = 2.2.4
* tensorflow = 1.8.0
* pyAudioAnalysis = 0.2.5
* librosa = 0.6.2
* scipy = 1.1.0

## Usage
Since the function "stFeatureSpeed" in pyAudioAnalysis is default unworkable, you have to modify the code in audioFeatureExtraction.py (for index related issue, just cast the value type to integer; for the issue in method stHarmonic, cast M to integer(M = int(M); Comment out the invocation of method 'mfccInitFilterBanks' in stFeatureSpeed).

```
from predict import load_model,get_audioclass

Load model: 
model = load_model('model/best_model.h5')

get_result: 
predict_class,predict_prob = get_audioclass(model,wav_file_path)

get_allresult:  
# result_dic: {class:prob}
predict_class,predict_prob,result_dic = get_audioclass(model,wav_file_path,all = True)

```

## References
* https://github.com/RayanWang/Speech_emotion_recognition_BLSTM
* Li, D., Mei, H., Shen, Y., Su, S., Zhang, W., Wang, J., ... & Chen, W. (2018). ECharts: A declarative framework for rapid construction of web-based visualization. Visual Informatics.
