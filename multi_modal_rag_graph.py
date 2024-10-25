from make_multi_vectorDB.multi_vector_retriever import load_stores_and_create_context_graph
from chain.multimodal_chain_graph import multi_modal_rag_chain_graph
import yaml
from graph_state.graph_state_chain import GraphState
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

with open('config.yaml', 'r') as f:
    config = yaml.full_load(f)

import warnings
warnings.filterwarnings("ignore")

workflow = StateGraph(GraphState)

workflow.add_node("load_multi-vector-retriever",load_stores_and_create_context_graph)
workflow.add_node("multimodal_rag", multi_modal_rag_chain_graph)

workflow.add_edge("load_multi-vector-retriever","multimodal_rag")

workflow.set_entry_point("load_multi-vector-retriever")


memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

graph_png = app.get_graph(xray=True).draw_mermaid_png()

# 파일로 저장
with open("rag_graph_output.png", "wb") as f:
    f.write(graph_png)
 
graph_config = RunnableConfig(recursion_limit=20, configurable={"thread_id": "Multi-RAG-Answer"})

query = "RAFT가 무엇인가요?"

inputs = GraphState(query=query,docstore_path=config["store"]["docstore_path"] ,vectorstore_directory=config["store"]["vectorstore_directory"])
output = app.invoke(inputs,config=graph_config)

print(output["answer"])