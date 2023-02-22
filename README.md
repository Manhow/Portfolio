# Portfolio
Hello, here is a collection of my portfolio projects that showcase my skills in a hands-on manner:
1. [**k-nrm**](https://arxiv.org/pdf/1706.06613.pdf)

This project is a small-scale web service built with Flask, aimed at discovering and presenting similar questions. The system takes a list of queries as input, calculates the top 100 candidates for each question from the [faiss](https://github.com/facebookresearch/faiss) index, then ranks them using k-nrm and outputs the top k most relevant candidates. [GloVe](https://nlp.stanford.edu/projects/glove/) vectors were used as text embeddings. The k-nrm model was trained on the widely used [Quora Question Pairs (QQP) dataset](https://gluebenchmark.com/tasks) dataset.
The ranking model demonstrated impressive accuracy, with a [NDCG](https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1) score of approximately 0.93 on validation data, even without retraining the embeddings. K-nrm is incredibly fast and efficient, requiring only a small number of kernel and bias parameters (21 + 1 in this case) as compared to models with millions of parameters. The most time-consuming task is calculating cosine similarity matrices, and the model can be trained even without a GPU. By adjusting the mean and variance values of the kernels, the sensitivity of the model can be fine-tuned to better match candidates (see article). This model utilizes transfer learning as the architecture of k-nrm allows to retrain embeddings for your needs with kernel trick, so you can differentiate the whole algorithm, count gradients and pass them on input embeddings. The Flask application is versatile and can utilize various pre-trained embeddings, such as ([fasttext](https://fasttext.cc/), etc), by retrieving word indices from a dictionary and combining them with the embeddings. To do this, a custom dictionary and embedding matrix must be created.

For future research, you may want to consider studying the article [ColBERT](https://arxiv.org/pdf/2004.12832.pdf), which combines the concepts of k-nrm and [BERT](https://arxiv.org/pdf/1810.04805.pdf) models. It is noteworthy and quite impressive.

2. [**Uplift tree**](https://link.springer.com/content/pdf/10.1007/s10115-011-0434-0.pdf)

The segmentation of consumers according to the net effect of marketing impact allows a business to focus the advertising budget on customers who are ready to carry out the target action only if there is communication.


This class implements uplift tree with DeltaDeltaP criterion, contains classic fit and predict methods. The input receives features X, the target variable Y, treatment/control flag T. Features are calculated for the period before the company’s running, target variable for the promotion period. Start promotion is accompanied by sending sms or push notifications

Keep in mind, the presence of a single tree suggests the existence of a dense forest...Build a gradient boosting model!
(https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775)
(https://github.com/uber/causalml/blob/master/causalml/inference/tree/causal/causaltree.py)
