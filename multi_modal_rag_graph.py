from make_multi_vectorDB.multi_vector_retriever import load_stores_and_create_multivectorRetriever
from chain.multimodal_chain import multi_modal_rag_chain
from make_multi_vectorDB.image_utils import plt_img_base64
from langgraph.graph import END, StateGraph

import warnings
warnings.filterwarnings("ignore")

workflow = StateGraph(GraphState)
def execute_chain(multi_vector_retriever, query):
    chain = multi_modal_rag_chain(multi_vector_retriever)
    return chain.invoke(query)

load_stores_node = GraphNode(
    name="load_stores",
    func=load_stores_and_create_multivectorRetriever,
    inputs={"vectorstore_directory": "./store/vectorstore/multi_modal_data", "docstore_path": "./store/docstore/docstore.pkl"},
)

execute_chain_node = GraphNode(
    name="execute_chain",
    func=execute_chain,
    inputs={"query": "RAFT가 무엇인가요?"},
)

# 노드 간의 edge 연결
# load_stores_node의 출력인 multi_vector_retriever를 execute_chain_node의 입력으로 연결
edge = Edge(
    source=load_stores_node,
    source_output="output",  # load_stores_node의 출력
    destination=execute_chain_node,
    destination_input="multi_vector_retriever"  # execute_chain_node의 입력으로 연결
)
graph = Graph(nodes=[load_stores_node, execute_chain_node], edges=[edge])
state = GraphState()

result = graph.run(state)

# 결과 출력
print(result["execute_chain"])
