import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.full_load(f)

def create_multi_vector_retriever(
        vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    요약을 색인화하지만 원본 이미지나 텍스트를 반환하는 검색기를 생성합니다.
    """
    # 저장 계층 초기화
    store = InMemoryStore()
    id_key = "doc_id"

    # 멀티 벡터 검색기 생성
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 문서를 벡터 저장소와 문서 저장소에 추가하는 헬퍼 함수
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [
            str(uuid.uuid4()) for _ in doc_contents
        ]  # 문서 내용마다 고유 ID 생성
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(
            summary_docs
        )  # 요약 문서를 벡터 저장소에 추가
        retriever.docstore.mset(
            list(zip(doc_ids, doc_contents))
        )  # 문서 내용을 문서 저장소에 추가

    # 텍스트, 테이블, 이미지 추가
    if text_summaries:
        #add_documents(retriever, text_summaries, texts)
        add_documents(retriever, text_summaries, list(texts.values()))  # texts를 리스트로 변환

    if table_summaries:
        #add_documents(retriever, table_summaries, tables)
        add_documents(retriever, table_summaries, list(tables.values()))  # tables를 리스트로 변환

    if image_summaries:
        #add_documents(retriever, image_summaries, images)
        add_documents(retriever, image_summaries, list(images.values()))  # images를 리스트로 변환

    return retriever
# 이거 분류하기
def add_documents_to_stores_and_save(
        vectorstore, docstore, text_summaries, texts, table_summaries, tables, image_summaries, images, id_key="doc_id"
):
    """
    동일한 ID로 벡터 스토어와 문서 저장소에 문서를 추가하고, 이를 저장합니다.
    """
    def add_documents(vectorstore, docstore, doc_summaries, doc_contents):
        # 문서마다 고유 ID 생성
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

        # 요약된 문서를 벡터 스토어에 추가
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        vectorstore.add_documents(summary_docs)

        # 동일한 ID로 원본 문서를 docstore에 추가
        docstore.mset(list(zip(doc_ids, doc_contents)))

    # 텍스트, 테이블, 이미지 데이터 추가 (state에서 가져옴)
    if text_summaries:
        add_documents(vectorstore, docstore, list(text_summaries.values()), list(texts.values()))
    if table_summaries:
        add_documents(vectorstore, docstore, list(table_summaries.values()), list(tables.values()))
    if image_summaries:
        add_documents(vectorstore, docstore, list(image_summaries.values()), list(images.values()))

    # 벡터 스토어 저장 (자동저장)

    # docstore 저장 (pickle)
    with open(config["store"]["docstore_path"], "wb") as f:
        pickle.dump(docstore, f)  # docstore의 내용을 파일로 저장


def load_stores_and_create_multivectorRetriever(vectorstore_directory="./vectorstore/multi_modal_data", docstore_path="./store/docstore/docstore.pkl"):
    """
    저장된 벡터 스토어와 docstore를 불러와 retriever를 생성합니다.
    """
    hf_embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs={'device':'cuda'})

    # 벡터 스토어 로드 (Chroma 사용)
    vectorstore = Chroma(
        collection_name="multi-modal-rag",
        embedding_function=hf_embeddings,
        persist_directory=vectorstore_directory  # 저장된 벡터 스토어 위치
    )

    # 저장된 docstore 로드 (pickle 사용)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)

    # 멀티 벡터 검색기 생성
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,  # 로드된 docstore 사용
        id_key="doc_id"
    )

    return retriever
from graph_state.graph_state_chain import GraphState

def load_stores_and_create_context_graph(state:GraphState):
    vectorstore_directory=state["vectorstore_directory"]
    docstore_path=state["docstore_path"]
    
    """
    저장된 벡터 스토어와 docstore를 불러와 retriever를 생성합니다.
    """
    hf_embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs={'device':'cuda'})

    # 벡터 스토어 로드 (Chroma 사용)
    vectorstore = Chroma(
        collection_name="multi-modal-rag",
        embedding_function=hf_embeddings,
        persist_directory=vectorstore_directory  # 저장된 벡터 스토어 위치
    )

    # 저장된 docstore 로드 (pickle 사용)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)

    # 멀티 벡터 검색기 생성
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,  # 로드된 docstore 사용
        id_key="doc_id"
    )
    context = retriever.get_relevant_documents(state["query"])

    return GraphState(context=context)