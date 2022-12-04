from typing import Dict, List, Tuple, Union, Callable
import os
import string
import json

from flask import Flask, jsonify, request, abort
import torch
import torch.nn.functional as F
import numpy as np
import faiss
import nltk
from langdetect import detect

app = Flask(__name__)


class GaussianKernel(torch.nn.Module):

    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x - self.mu)**2 / (2 * self.sigma**2))


class KNRM(torch.nn.Module):

    def __init__(self,
                 embedding_matrix: np.ndarray,
                 freeze_embeddings: bool,
                 kernel_num: int = 21,
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001,
                 out_layers: List[int] = []):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0)

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (self.kernel_num -
                                                          1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(torch.nn.Linear(in_f, out_f), torch.nn.ReLU())
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor],
                input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor,
                             doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())
        # shape = [B, L, R]
        matching_matrix = torch.einsum('bld,brd->blr',
                                       F.normalize(embed_query, p=2, dim=-1),
                                       F.normalize(embed_doc, p=2, dim=-1))
        return matching_matrix

    def _apply_kernels(
            self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out


class Solution:

    def __init__(self):
        self.EMB_PATH_KNRM = os.environ.get('EMB_PATH_KNRM')
        self.VOCAB_PATH = os.environ.get('VOCAB_PATH')
        self.EMB_PATH_GLOVE = os.environ.get('EMB_PATH_GLOVE')
        self.MLP_PATH = os.environ.get('MLP_PATH')
        self.embeddings, self.vocab, self.embeds_raw, self.mlp_weights = self._load_data(
        )

        self.model = KNRM(self.embeddings,
                          freeze_embeddings=True,
                          kernel_num=21)
        self.model.mlp.load_state_dict(self.mlp_weights)

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        glove_model = {}
        with open(file_path, 'r', encoding="utf8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        return glove_model

    def _load_data(self):
        # read embeddings from KNRM model
        embeddings = torch.load(self.EMB_PATH_KNRM)['weight']
        # read vocab
        with open(self.VOCAB_PATH, 'rb') as f:
            vocab = json.load(f)
        # read raw embeddings
        embeds_raw = self._read_glove_embeddings(self.EMB_PATH_GLOVE)
        # get mlp
        mlp_weights = torch.load(self.MLP_PATH)
        return embeddings, vocab, embeds_raw, mlp_weights

    def hadle_punctuation(self, inp_str: str) -> str:
        table = str.maketrans(string.punctuation, ' ' * 32)
        return inp_str.translate(table)

    def simple_preproc(self, inp_str: str) -> List[str]:
        new_str = self.hadle_punctuation(inp_str).lower()
        return nltk.word_tokenize(new_str)

    def _create_sentence_embedding(self, input_text: list,
                                   size: int) -> np.ndarray:
        # Sum (or mean) approach
        sent_embeds_matrix = np.zeros((len(input_text), size))
        res = [self.simple_preproc(val) for val in list(input_text)]
        for i, sentence in enumerate(res):
            l = [
                self.embeds_raw[val] for val in sentence
                if val in self.embeds_raw.keys()
            ]  # list of lists with embeds
            sent_embeds_matrix[i, :] = np.sum(l, axis=0)
        return np.float32(sent_embeds_matrix)

    def _get_index(self, texts: Dict[str, str]):
        self.size = 50
        self.keys_mapping = list(texts.keys())
        self.candidate_text = list(texts.values())
        sent_embeds_matrix = self._create_sentence_embedding(
            self.candidate_text, self.size)
        # cosine
        self.index = faiss.index_factory(self.size, "Flat",
                                         faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(sent_embeds_matrix)
        self.index.add(sent_embeds_matrix)
        pass

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        res = [self.vocab.get(i, self.vocab['OOV']) for i in tokenized_text]
        return res

    def _convert_text_idx_to_token_idxs(self, curr_text) -> List[int]:
        tokenized_text = self.simple_preproc(curr_text)
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs


sol = Solution()


@app.route("/ping", methods=['GET'])
def ping():
    if hasattr(sol, 'model'):
        return jsonify(status='ok')
    else:
        return jsonify(status='failure')


@app.route("/update_index", methods=['POST'])
def update_index():
    data = json.loads(request.json)  # Dict[str,str]
    texts = dict(data['documents'])
    sol._get_index(texts)
    if sol.index.ntotal > 0:
        return jsonify(status='ok', index_size=sol.index.ntotal)
    else:
        return jsonify(status='FAISS failure...')


@app.route("/query", methods=['POST'])
def query():
    if hasattr(sol, 'index') == False:
        return jsonify(status='FAISS is not initialized!')
    else:
        data = json.loads(request.json)  # Dict['queries', List[str]]
        queries = data['queries']
        # search neighbours
        k = 100
        search_sent_prepared = sol._create_sentence_embedding(
            queries, sol.size)
        faiss.normalize_L2(search_sent_prepared)
        # sol.index.nprobe = 5
        _, I = sol.index.search(search_sent_prepared, k)
        suggested_sents = []
        for i in range(I.shape[0]):
            tmp = []
            tmp_keys = []
            for j in range(len(I[i])):
                tmp.append(sol.candidate_text[I[i, j]])
                tmp_keys.append(sol.keys_mapping[I[i, j]])
            suggested_sents.append(tmp)
        # KNRM
        dict_len = 30
        total_predictors = []
        bool_list = []

        for i in range(len(queries)):
            lang_check = detect(queries[i])
            if lang_check != 'en':
                bool_list.append(False)
                total_predictors.append(None)
            else:
                bool_list.append(True)
                pair = {}
                query_matrix = torch.zeros(size=(len(suggested_sents[i]),
                                                 dict_len))
                doc_matrix = torch.zeros(size=(len(suggested_sents[i]),
                                               dict_len))
                for j in range(len(suggested_sents[i])):
                    query_idxs = sol._convert_text_idx_to_token_idxs(
                        queries[i])[:dict_len]
                    doc_idxs = sol._convert_text_idx_to_token_idxs(
                        suggested_sents[i][j])[:dict_len]
                    if len(query_idxs) < dict_len:
                        query_idxs = np.pad(
                            query_idxs,
                            (0, dict_len - len(query_idxs) % sol.size),
                            'constant')
                    if len(doc_idxs) < dict_len:
                        doc_idxs = np.pad(
                            doc_idxs, (0, dict_len - len(doc_idxs) % sol.size),
                            'constant')
                    query_matrix[j, :] = torch.FloatTensor(query_idxs)
                    doc_matrix[j, :] = torch.FloatTensor(doc_idxs)
                pair = {'query': query_matrix, 'document': doc_matrix}

                y_preds = sol.model.predict(pair)

                _, argsort = torch.sort(y_preds, descending=True, dim=0)
                sorted_sentences = [suggested_sents[i][k]
                                    for k in argsort][:10]
                temp_list = [(sol.keys_mapping[sol.candidate_text.index(val)],
                              val) for val in sorted_sentences]
                total_predictors.append(temp_list)

        return jsonify(lang_check=bool_list, suggestions=total_predictors)


if __name__ == "__main__":
    app.run(debug=True)