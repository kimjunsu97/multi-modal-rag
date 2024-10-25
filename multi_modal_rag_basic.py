from make_multi_vectorDB.multi_vector_retriever import load_stores_and_create_multivectorRetriever
from chain.multimodal_chain import multi_modal_rag_chain
from make_multi_vectorDB.image_utils import plt_img_base64
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.full_load(f)

import warnings
warnings.filterwarnings("ignore")

multi_vector_retriever = load_stores_and_create_multivectorRetriever(vectorstore_directory=config["store"]["vectorstore_directory"],
                                                                      docstore_path=config["store"]["docstore_path"])
chain_multimodal_rag = multi_modal_rag_chain(multi_vector_retriever)

query = "RAFT가 무엇인가요?"
#query = "Introduction 요약해줘"
print(chain_multimodal_rag.invoke(query))

docs = multi_vector_retriever.invoke(query, limit=6)
print(len(docs))
