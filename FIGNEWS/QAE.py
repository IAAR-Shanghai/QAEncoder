import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import math
import json
from tqdm import tqdm
import random
import pandas as pd
from typing import Any, List
from config import *
from tool import *
import numpy as np
from loguru import logger
import logging
from concurrent.futures import ThreadPoolExecutor
import argparse
from sentence_transformers import SentenceTransformer
from datetime import datetime

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index import download_loader
from llama_index.core.embeddings.base import BaseEmbedding
from llama_index import ServiceContext, StorageContext
from llama_index.vector_stores import MilvusVectorStore, FaissVectorStore
from llama_index.bridge.pydantic import PrivateAttr, Field
from llama_index import Document, QueryBundle
from llama_index.schema import TextNode
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import faiss

class MyDataset():
    def __init__(self, configs):
        dataset_name = configs['dataset_name']
        if 'fig' in dataset_name:
            language = dataset_name.split('fig')[1]
            configs['pool_path'] = f"./data/pool/{language}_pool.jsonl"
            configs['benchmark_path_list'] = [f"./data/benchmark/{language}_benchmark.jsonl"]
            configs['query_path'] = f"./data/query/{language}/"
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

class QAEmbedding(BaseEmbedding):
    _model: Any = PrivateAttr()
    _mode: str = PrivateAttr()
    _configs: Any = PrivateAttr()
    
    def __init__(self, configurations):
        self._configs = configurations
        self._model = SentenceTransformer(self._configs["embed_model_path"], trust_remote_code=True)
        self._mode = "Chunk"
        super().__init__()
        if self._configs["encode_embed"]:
            self._write_chunk_embed()

    @classmethod
    def class_name(cls) -> str:
        return "QAEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _read_chunk_embed(self, text):
        try:
            text_id, text = text.split('\n\n')
            text_id = text_id.split(': ')[1]
            embed = load_data_abs(self._configs['embed_path']+text_id)
            if embed is None:
                raise ValueError(f"Embedding not found for {text_id}")
            if len(embed) != self._configs['embed_dim']:
                raise ValueError(f"Invalid embedding length: {len(embed)}")
            return embed
        except ValueError as e:
            logging.error(e)
            return np.zeros(self._configs['embed_dim'])

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._mode == "Chunk":
            with ThreadPoolExecutor(max_workers=self._configs['max_thread_num']) as executor:
                QAE_embeddings = list(executor.map(self._read_chunk_embed, texts))
        elif self._mode == "Query":
            QAE_embeddings = self._model.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")
        return QAE_embeddings

    def _write_chunk_embed_single_QAE_emb(self, text, questions):
        text = text.strip()
        for i in range(len(questions)):
            try:
                questions[i] = questions[i].split('. ')[1].strip()
            except Exception as e:
                pass
        embeds =  self._model.encode([text] + questions, normalize_embeddings=True)
        old_embed, new_embeds = embeds[0], embeds[1:]
        new_embed = new_embeds.mean(axis=0)
        new_embed = new_embed / np.linalg.norm(new_embed)
        com_embed = self._configs['alpha_value']*new_embed + (1-self._configs['alpha_value'])*old_embed
        com_embed = com_embed / np.linalg.norm(com_embed)
        return com_embed
    
    def _write_chunk_embed_single_QAE_txt(self, text, questions):
        text = text.strip()
        for i in range(len(questions)):
            try:
                questions[i] = questions[i].split('. ')[1].strip()
            except Exception as e:
                pass
        concates = []
        for i in range(len(questions)):
            idxes = list(range(len(questions)))
            random.shuffle(idxes)
            ques = ''
            for j in idxes:
                ques = ques + ' ' + questions[j]
                if len(ques) >= len(text)*self._configs['beta_value']:
                    break
            concates.append(text + ques)
        concates_embed = self._model.encode(concates, normalize_embeddings=True)
        concates_embed = concates_embed.mean(axis=0)
        concates_embed = concates_embed/np.linalg.norm(concates_embed)
        return concates_embed
    
    def _write_chunk_embed_single_QAE_hyb(self, text, questions):
        text = text.strip()
        for i in range(len(questions)):
            try:
                questions[i] = questions[i].split('. ')[1].strip()
            except Exception as e:
                pass
        concates = []
        for i in range(len(questions)):
            idxes = list(range(len(questions)))
            random.shuffle(idxes)
            ques = ''
            for j in idxes:
                ques = ques + ' ' + questions[j]
                if len(ques) >= len(text)*self._configs['beta_value']:
                    break
            concates.append(text + ques)
        embeds = self._model.encode(concates + questions, normalize_embeddings=True)
        concates_embed, ques_embed = embeds[:len(concates)], embeds[len(concates):]
        concates_embed = concates_embed.mean(axis=0)
        concates_embed = concates_embed/np.linalg.norm(concates_embed)
        ques_embed = ques_embed.mean(axis=0)
        ques_embed = ques_embed/np.linalg.norm(ques_embed)

        com_embed = self._configs['alpha_value']*ques_embed + (1-self._configs['alpha_value'])*concates_embed
        com_embed = com_embed / np.linalg.norm(com_embed)
        return com_embed
    
    def _write_chunk_embed_single(self, text, questions):
        if self._configs['method'] == "QAE_emb":
            return self._write_chunk_embed_single_QAE_emb(text, questions)
        if self._configs['method'] == "QAE_txt":
            return self._write_chunk_embed_single_QAE_txt(text, questions)
        if self._configs['method'] == "QAE_hyb":
            return self._write_chunk_embed_single_QAE_hyb(text, questions)

    def _write_chunk_embed_single_cached(self, chunk):
        chunk_id, text = chunk
        doc_id = self._configs['query_path'] + chunk_id
        embed_id = self._configs['embed_path'] + chunk_id
        try:
            if(check_data_abs(embed_id)):
                return True
            while(not check_data_abs(doc_id)):
                return False
            questions = load_data_abs(doc_id)
            embeds = self._write_chunk_embed_single(text, questions)
            dump_data_abs(embeds, embed_id)
        except Exception as e:
            return False
        return True
    
    def _write_chunk_embed(self):
        chunks = []
        with open(self._configs['pool_path'], "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                chunks.append(data)
        with ThreadPoolExecutor(max_workers=self._configs['max_thread_num']) as executor:
            list(tqdm(executor.map(self._write_chunk_embed_single_cached, chunks), total=len(chunks)))

    def chunk_mode(self):
        self._mode = "Chunk"
    
    def query_mode(self):
        self._mode = "Query"
    
class QAEncoder():     
    def __init__(self, configurations):
        self.configs = configurations
        self.embed_model = QAEmbedding(self.configs)
        if self.configs['use_reranker']:
            self.reranker_model = FlagEmbeddingReranker(top_n=self.configs['top_k'], model=self.configs['reranker_model_path'])
        self.vector_index = None
        if configurations["construct_index"]:
            self.build_index()
        else:
            self.load_index()
        self.embed_model.query_mode()
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.configs['top_k'],
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def load_index(self):
        raise NotImplementedError("Loading index is not implemented yet!")
    
    def build_index(self):
        files = os.listdir(self.configs['embed_path'])
        with open(self.configs['embed_path']+files[0], "rb") as f:
            data = pickle.load(f)
            self.configs['embed_dim'] = len(data)
        
        nodes = []
        with open(self.configs['pool_path'], "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                chunk_id, text = data
                if check_data_abs(self.configs['query_path']+chunk_id):
                    nodes.append(TextNode(text=text, extra_info={'chunk_id':chunk_id}))

        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        faiss_index = faiss.IndexFlatIP(self.configs['embed_dim'])
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.vector_index = GPTVectorStoreIndex(
            nodes, service_context=service_context, 
            storage_context=storage_context, show_progress=True
        )
        print(f"Indexing {self.configs['collection_name']} finished!")
    
    def search_docs(self, query_text):
        response_vector = self.query_engine.retrieve(QueryBundle(query_str=query_text))
        if self.configs['use_reranker']:
            response_vector = self.reranker_model._postprocess_nodes(response_vector, QueryBundle(query_str=query_text))
        return response_vector

    def evaluate(self):
        result_list = []
        random.seed(self.configs["seed"])
        for benchmark_path in self.configs['benchmark_path_list']:
            print(f"Start evaluating {benchmark_path}!")
            question_with_provenance = []
            with open(benchmark_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    question_with_provenance.append(data)
            # random.shuffle(question_with_provenance)
            question_with_provenance = question_with_provenance[:self.configs['benchmark_num']]
            with ThreadPoolExecutor(max_workers=self.configs['max_thread_num']) as executor:
                scores = list(tqdm(executor.map(self.evaluate_single, question_with_provenance), total=len(question_with_provenance)))
                mrr_sum, map_sum, ndcg_sum, precision_sum, recall_sum, f1_sum = 0, 0, 0, 0, 0, 0
                hit_sum = 0
                sim_sum = 0
                for mrr_score, map_score, ndcg_score, precision_score, recall_score, f1_score, sim_score in scores:
                    mrr_sum += mrr_score
                    map_sum += map_score
                    ndcg_sum += ndcg_score
                    precision_sum += precision_score
                    recall_sum += recall_score
                    f1_sum += f1_score
                    sim_sum += sim_score
                    if mrr_score > 0:
                        hit_sum += 1
                mrr = mrr_sum/len(question_with_provenance)
                map = map_sum/len(question_with_provenance)
                ndcg = ndcg_sum/len(question_with_provenance)
                precision = precision_sum/len(question_with_provenance)
                recall = recall_sum/len(question_with_provenance)
                f1 = f1_sum/len(question_with_provenance)
                sim = sim_sum/len(question_with_provenance)
                hit_rate = hit_sum/len(question_with_provenance)
                print(f"MRR: {round(mrr, 4)}, MAP: {round(map, 4)}, NDCG: {round(ndcg, 4)}, Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, F1: {round(f1, 4)}, Sim: {round(sim, 4)}, Hit Rate: {round(hit_rate, 4)}")
                obj = {
                    "MRR": round(mrr, 4),
                    "MAP": round(map, 4),
                    "NDCG": round(ndcg, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1": round(f1, 4),
                    "Sim": round(sim, 4),
                    "Hit Rate": round(hit_rate, 4),
                    "configurations": self.configs
                }
                result_list.append(obj)
        with open(self.configs['output_path'], "a") as f: 
            json.dump(result_list, f, indent=4, ensure_ascii=False)

    def evaluate_single(self, question_with_provenance):
        question, provenance = question_with_provenance['question'], question_with_provenance['provenance']
        nodes_with_score = self.search_docs(question)
        return self._get_score(provenance, nodes_with_score)

    def _get_score(self, provenance, nodes_with_score):
        mrr_score = self._mrr_score(provenance, nodes_with_score)
        map_score = self._map_score(provenance, nodes_with_score)
        ndcg_score = self._ndcg_score(provenance, nodes_with_score)
        precision_score = self._precision_score(provenance, nodes_with_score)
        recall_score = self._recall_score(provenance, nodes_with_score)
        epison = 1e-6
        f1_score = 2*precision_score*recall_score/(precision_score+recall_score+epison)
        score_avg = 0
        for node in nodes_with_score:
            score_avg += node.score
        score_avg = score_avg/len(nodes_with_score)
        return mrr_score, map_score, ndcg_score, precision_score, recall_score, f1_score, score_avg

    def _mrr_score(self, provenance, nodes_with_score):
        sum = 0
        chunk_id_set = set()
        for idx, node_with_score in enumerate(nodes_with_score):
            chunk_id = node_with_score.metadata['chunk_id']
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            if chunk_id in provenance:
                sum += 1/(idx+1)
                return sum
        return sum

    def _map_score(self, provenance, nodes_with_score):
        pos = 0
        cnt = 0
        tot_precision = 0
        chunk_id_set = set()
        for idx, node_with_score in enumerate(nodes_with_score):
            chunk_id = node_with_score.metadata['chunk_id']
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            cnt += 1
            if chunk_id in provenance:
                pos += 1
                tot_precision += pos/cnt
        if pos == 0:
            return 0
        avg_precision = tot_precision/pos
        return avg_precision

    def _ndcg_score(self, provenance, nodes_with_score):
        dcg = 0
        pos = 0
        chunk_id_set = set()
        for idx, node_with_score in enumerate(nodes_with_score):
            chunk_id = node_with_score.metadata['chunk_id']
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            if chunk_id in provenance:
                pos += 1
                dcg += 1/math.log2(idx+2)
        idcg = 0
        for i in range(len(provenance)):
            idcg += 1/math.log2(i+2)
        if idcg == 0:
            return 0
        return dcg/idcg

    def _precision_score(self, provenance, nodes_with_score):
        sum = 0
        chunk_id_set = set()
        for idx, node_with_score in enumerate(nodes_with_score):
            chunk_id = node_with_score.metadata['chunk_id']
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            if chunk_id in provenance:
                sum += 1
        return sum/len(nodes_with_score)

    def _recall_score(self, provenance, nodes_with_score):
        sum = 0
        chunk_id_set = set()
        for idx, node_with_score in enumerate(nodes_with_score):
            chunk_id = node_with_score.metadata['chunk_id']
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            if chunk_id in provenance:
                sum += 1
        return sum/len(provenance)
    
if __name__ == '__main__':
    description = "QAEncoder Base File"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default=description)
    parser.add_argument('--method', type=str, default="QAE_emb")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    parser.add_argument('--alpha_value', type=float, default=0.0)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--beta_value', type=float, default=1.0)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--embed_model_path', type=str, default="./model/bge-large-en-v1.5/")
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--benchmark_num', type=int, default=500)
    parser.add_argument('--max_thread_num', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default="figEnglish")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encode_embed', type=bool, default=True)
    parser.add_argument('--construct_index', type=bool, default=True)
    parser.add_argument('--use_reranker', type=bool, default=False)
    parser.add_argument('--reranker_model_path', type=str, default="xxx")
    # parser.add_argument('--devices', type=list, default=[0,1,2,3,4,5,6,7])
    configs = vars(parser.parse_args())
    mydataset = MyDataset(configs)

    if configs['method'] == "QAE_emb":
        configs['params'] = str(int(configs['alpha_value']*100))
    if configs['method'] == "QAE_txt":
        configs['params'] = str(int(configs['beta_value']*100))
    if configs['method'] == "QAE_hyb":
        configs['params'] = str(int(configs['beta_value']*100))+'_'+str(int(configs['alpha_value']*100))

    configs['model_id'] = configs['embed_model_path'].split('/')[-2]
    configs['model_id'] = configs['model_id'].replace('-', '_').replace('.', '_')
    configs['collection_name'] = f"{configs['model_id']}_{configs['method']}_{configs['params']}"
    configs['embed_path'] = configs['query_path'] + configs['collection_name'] + '/'
    if not os.path.exists(configs['embed_path']):
        os.makedirs(configs['embed_path'])
    output_dir = f"./output/{configs['dataset_name']}/"    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    configs['output_path'] = output_dir+configs['collection_name']+f"_{configs['timestamp']}.json"
    random.seed(configs["seed"])

    logging.basicConfig(filename = output_dir+'error.log', level=logging.ERROR)
    
    # check duplicate
    base_name = output_dir + configs['collection_name']
    files = os.listdir(output_dir)
    for file in files:
        file_base_name = '_'.join(file.split('_')[:-1])
        if file_base_name == base_name:
            with open(output_dir+file, 'r') as f:
                text = f.read()
                data = json.loads(text)
                if len(data) >= len(configs['benchmark_path_list']):
                    exit(0)
    # check duplicate

    logger.info(f"Start running with configurations: {configs}")
    qa_encoder = QAEncoder(configs)
    qa_encoder.evaluate()