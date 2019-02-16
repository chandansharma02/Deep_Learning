# Deep_Learning
Deep learning code repository

Cyclic learning rate: https://arxiv.org/pdf/1506.01186.pdf

It is known that the learning rate is the most important hyper-parameter to tune for training deep neural networks. This paper describes a new method for setting the learning rate, named cyclical learning rates, which practically eliminates the need to experimentally ﬁnd the best values and schedule for the global learning rates. 

References

model structure & clr from https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr
hidden size 256 from https://www.kaggle.com/artgor/text-modelling-in-pytorch
speed up pre-processing from https://www.kaggle.com/syhens/speed-up-your-preprocessing
the idea to reduce oov from https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
misspell dictionary & punctuations from https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
latex cleaning from https://www.kaggle.com/sunnymarkliu/more-text-cleaning-to-increase-word-coverage
pytorch text processing routines from https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
capsule from https://www.kaggle.com/spirosrap/bilstm-attention-kfold-clr-extra-features-capsule
DeepMoji from https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
performance
fast geometric ensemble from https://arxiv.org/abs/1802.10026. It gives a consistant and significant boost in both LB score and CV score for various models when combined with a learnable embedding.
semi-supervised ensemble similar to Malware Classification Challenge 1st solution. Marginal significance can be observed with a large test set. it doesn't bring me any benefits in the 2nd stage.
"mix up" embeddings. The idea is to randomly choose a linear combination between two embeddings rather than simple averaging. Though no significant improvement can be observed, I still keep it in my solution as regularization.
speed
bucket iterator. similar to the one in torchtext. It runs twice as fast as static padding.
miscs
speed up capsule. 
load embedding file with pandas. It saves ~80 seconds per embedding. 
reduce oov by replacing oov word with its capitized, upper, lower version if available. The final oov rate is about 7.5%. 
