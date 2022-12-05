# Portfolio
Hi there, here you can find my portfolio projects that will allow to reveal my skills in a practical way:
1. [**k-nrm**](https://arxiv.org/pdf/1706.06613.pdf)

This project is a micro web service to find similar questions based on Flask. At the input a list of queries is submitted and for each of the questions the top 100 candidates from the [faiss](https://github.com/facebookresearch/faiss) index are calculated, then k-nrm ranks them and gives top k of the most relevant candidates. [GloVe](https://nlp.stanford.edu/projects/glove/) vectors were used as text embeddings. The popular [Quora Question Pairs (QQP) dataset](https://gluebenchmark.com/tasks) was used for training k-nrm. 
Even without the training of the embeddings, the ranking model showed NDCG ≈ 0.93 on the validation data, which indicates a high accuracy of the model under consideration. Instead of models with many millions of parameters, k-nrm offers only a number of kernels + bias parameters (in my case 21 + 1), which makes it incredibly fast (the most difficult to calculate cosine similarity matrices).

