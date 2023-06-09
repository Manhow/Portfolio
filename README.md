# Portfolio
Hello, here is a collection of my portfolio projects that showcase my skills in a hands-on manner:
1. [**k-nrm**](https://arxiv.org/pdf/1706.06613.pdf)

This project is a small-scale web service built with Flask, aimed at discovering and presenting similar questions. The system takes a list of queries as input, calculates the top 100 candidates for each question from the [faiss](https://github.com/facebookresearch/faiss) index, then ranks them using k-nrm and outputs the top k most relevant candidates. [GloVe](https://nlp.stanford.edu/projects/glove/) vectors were used as text embeddings. The k-nrm model was trained on the widely used [Quora Question Pairs (QQP) dataset](https://gluebenchmark.com/tasks) dataset.
The ranking model demonstrated impressive accuracy, with a [NDCG](https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1) score of approximately 0.93 on validation data, even without retraining the embeddings. K-nrm is incredibly fast and efficient, requiring only a small number of kernel and bias parameters (21 + 1 in this case) as compared to models with millions of parameters. The most time-consuming task is calculating cosine similarity matrices, and the model can be trained even without a GPU. By adjusting the mean and variance values of the kernels, the sensitivity of the model can be fine-tuned to better match candidates (see article). This model utilizes transfer learning as the architecture of k-nrm allows to retrain embeddings for your needs with kernel trick, so you can differentiate the whole algorithm, count gradients and pass them on input embeddings. The Flask application is versatile and can utilize various pre-trained embeddings, such as ([fasttext](https://fasttext.cc/), etc), by retrieving word indices from a dictionary and combining them with the embeddings. To do this, a custom dictionary and embedding matrix must be created.

For future research, you may want to consider studying the article [ColBERT](https://arxiv.org/pdf/2004.12832.pdf), which combines the concepts of k-nrm and [BERT](https://arxiv.org/pdf/1810.04805.pdf) models. It is noteworthy and quite impressive.

Usage tips. Put the [GLoVE](https://nlp.stanford.edu/projects/glove/) embeddings into data/ (170 Mbytes) to have docker add the file to the environment variable. Main.py can be used for creating your own vocab, triplets (train, test) and network training.

2. [**Uplift Tree**](https://link.springer.com/content/pdf/10.1007/s10115-011-0434-0.pdf)

There are four primary types of customers: those who make purchases regardless of communication, those who make purchases only after communication, those who do not make purchases regardless of communication, and those who do not make purchases even after communication. By segmenting consumers based on the net impact of marketing, a business can effectively allocate its advertising budget towards customers who are likely to take the desired action **only after communication**. This enables the business to maximize the impact of its marketing efforts and increase the likelihood of a successful outcome. It may be natural to consider calculating the causal effect for each buyer, but in fact, this cannot be done, since we cannot send an offer to one user and not send it at the same time. Rather than attempting to calculate the causal effect, let us instead focus on predicting the uplift of the variable Y: let's measure the extent to which the client's reaction changes in response to exposure, compared to their reaction without exposure. Roughly speaking, the average increase in Y when communicating. Our objective is to develop an uplift model that can accurately forecast the expected change in the client's response as a result of exposure. This class incorporates an uplift tree utilizing the DeltaDeltaP criterion, complete with conventional fit and predict methods. The input receives features X, the target variable Y, treatment/control flag T. The features are computed based on the period preceding the company's operation, while the target variable is determined during the promotional period. Start promotion is accompanied by sending sms or push notifications. Keep in mind, the presence of a single tree suggests the existence of a dense forest...Build a gradient boosting model! 

Additional information: [Basic decision tree methods](https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775), [pylift](https://pylift.readthedocs.io/en/latest/), [causalml](https://causalml.readthedocs.io/en/latest/about.html).

3. [**Feature Store**]
You've dedicated time to implement a feature calculation, only to discover it's already been done by a colleague within your team or a neighboring one. You've utilized this feature in model training, but due to the extensive computation time, it's not feasible for production use. Despite this, there remain a few areas where feature store could still be beneficial. The given code is about creating a flexible data processing pipeline using Dask for feature calculation and sklearn for transformation. 

Let's break down the main components of the code:
 * Engine: This class holds a dictionary of Dask DataFrames. You can add (register) new tables to it or retrieve existing ones.

 * FeatureCalcer: This is an abstract base class (ABC) which serves as a template for creating classes that calculate features. Any class derived from it needs to implement a compute method. It has a reference to an Engine instance to access data tables.

DateFeatureCalcer: This class extends FeatureCalcer and is initialized with a date. It does not implement the abstract compute method and should be extended further to do so.

CALCER_REFERENCE: This global dictionary is used to register feature calculator classes. You can add a new class to this dictionary using the register_calcer function, and then create an instance of a registered class using the create_calcer function.

compute_features: This function takes an Engine instance and a configuration dictionary, creates instances of feature calculators according to this configuration, calculates features using these calculators, and joins the results into a single DataFrame.

FunctionalTransformer: This is a custom sklearn transformer that takes any function as an argument and applies it to the data.

functional_transformer: This function is a helper to easily create instances of FunctionalTransformer.

TRANSFORMER_REFERENCE: This is another global dictionary used to register transformer classes or functions. Similar to CALCER_REFERENCE, you can add new transformers to this dictionary using the register_transformer function, and create instances of registered transformers using the create_transformer function.

build_pipeline: This function takes a configuration dictionary and builds an sklearn pipeline according to this configuration.
