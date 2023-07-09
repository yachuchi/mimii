# MIMII dataset baseline (Ver.1.0.3)
If you use the MIMII Dataset, please cite either of the following papers:

> [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019. URL: https://arxiv.org/abs/1909.09347

> [2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

## File description
```
baseline.py---------autoencoder code
baseline.yaml-------autoencoder parameter setting
requirements.txt----package to use MIMII dataset baseline
result--------------folder to save testing result
model---------------folder to save model loss
dataset-------------folder to save dataset
```
## Usage

###   Accuracy for all machines
Threshold: training average reconstruction error

|fan	|6dB		|0dB		|-6dB     |
|-------|---------------|---------------|-----------|
|id_00	|0.611793611793611	|0.615479115479115	|0.546683046683046  |
|id_02	|0.759052924791086	|0.688022284122562	|0.643454038997214|
|id_04	|0.775862068965517	|0.695402298850574	|0.527298850574712|
|id_06	|0.793628808864265	|0.793628808864265	|0.681440443213296 |

|pump	|6dB		|0dB		|-6dB	    |
|-------|---------------|---------------|-----------|
|id_00	|0.594405594405594	|0.555944055944055	|0.566433566433566|
|id_02	|0.391891891891891	| 0.481981981981982	|0.527027027027027|
|id_04	|0.5		|0.65	|0.735	    |
|id_06	|0.656862745098039	|0.759803921568627	|0.588235294117647  |

|slider	|6dB		|0dB		|-6dB     |
|-------|---------------|---------------|-----------|
|id_00	|0.720505617977528	|0.790730337078651	|0.778089888 |
|id_02	|0.743445692883895	|0.722846441947565	|0.691011235955056|
|id_04	|0.764044943820224	|0.775280898876404	|0.595505617977528|
|id_06	|0.741573033707865	|0.516853932584269	|0.516853932584269|

|valve	|6dB		|0dB		|-6dB     |
|-------|---------------|---------------|-----------|
|id_00  |0.579831932773109	|0.491596638655462	|0.508403361344537 |
|id_02  |0.6375	|0.5375	|0.541666666666666|
|id_04  |0.4125	|0.579166666666666	|0.495833333333333|
|id_06  |0.633333333333333	|0.570833333333333	|0.504166666666666|

-------------------------------------------------------------------
