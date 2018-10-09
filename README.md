# CMLDR

CMLDR, a novel drug repositioning computational method based on Collaborative Metric Learning (CML)[1], can recommend potential indications for known drugs having validated disease associations and new drugs without known associations. 
```
1. Dataset.
(1) Drug_simMat.txt store drug similarity matrix;
(2) DrDiAssMat.txt stores known drug-disease association information;
```
```
2. Code.
(1) MLDR-TN.py: predict potential indications for drugs;
(2) sampler.py : sample positive samples and negative samples.
(3) utils.py: split dataset into training, validation and test sets;
(4) evaluator.py: create evaluator for recall and precision evaluation;
```
[1] Hsieh, C. K. et al. (2017) Collaborative metric learning. In Proceedings of the 26th International Conference on World Wide Web, 193-201.
