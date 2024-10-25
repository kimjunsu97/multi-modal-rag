from graph_state.graph_state_manager import GraphState, initialize_graph_state, save_graph_state, load_graph_state
from make_multi_vectorDB.pdf_split import split_pdf
from make_multi_vectorDB.upstage_document_ai import analyze_layout
from make_multi_vectorDB.page_element import extract_page_metadata, extract_page_elements, extract_tag_elements_per_page, page_numbers
from make_multi_vectorDB.cropper import crop_image, generate_base64_image, crop_table
from make_multi_vectorDB.extract_text import extract_page_text, create_text_summary
from make_multi_vectorDB.extract_image import create_image_summary_data_batches, create_image_summary
from make_multi_vectorDB.extract_table import create_table_summary_data_batches, create_table_markdown, create_table_summary
from make_multi_vectorDB.multi_vector_retriever import add_documents_to_stores_and_save

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.full_load(f)

workflow = StateGraph(GraphState)

nodes = [
    ("split_pdf", split_pdf),
    ("analyze_layout", analyze_layout),
    ("extract_page_metadata", extract_page_metadata),
    ("extract_page_elements", extract_page_elements),
    ("extract_tag_elements_per_page", extract_tag_elements_per_page),
    ("get_page_numbers", page_numbers),
    ("crop_image", crop_image),
    ("generate_base64_image", generate_base64_image),
    ("crop_table", crop_table),
    ("extract_page_text", extract_page_text),
    ("create_text_summary", create_text_summary),
    ("create_image_summary_data_batches", create_image_summary_data_batches),
    ("create_table_summary_data_batches", create_table_summary_data_batches),
    ("create_image_summary", create_image_summary),
    ("create_table_summary", create_table_summary),
    ("create_table_markdown", create_table_markdown)
]

# for문으로 노드 순차적으로 추가
for name, function in nodes:
    workflow.add_node(name, function)

# # 순차적으로 노드 연결 (엣지 자동 추가)
# for i in range(1, len(nodes)):
#     workflow.add_edge(nodes[i-1][0], nodes[i][0])

workflow.add_edge("split_pdf","analyze_layout")
workflow.add_edge("split_pdf","analyze_layout")
workflow.add_edge("analyze_layout","extract_page_metadata")
workflow.add_edge("analyze_layout","extract_page_elements")
workflow.add_edge("extract_page_elements","extract_tag_elements_per_page")
workflow.add_edge("extract_tag_elements_per_page","get_page_numbers")

# 텍스트 + 요약 생성
workflow.add_edge("get_page_numbers","extract_page_text")
workflow.add_edge("extract_page_text","create_text_summary")

workflow.add_edge("create_text_summary","crop_image")
workflow.add_edge("create_text_summary","crop_table")

# 이미지 + 요약 생성
#workflow.add_edge("get_page_numbers","crop_image")
workflow.add_edge("crop_image","generate_base64_image")
workflow.add_edge("crop_image","create_image_summary_data_batches")
#workflow.add_edge("create_text_summary","create_image_summary_data_batches")
workflow.add_edge("create_image_summary_data_batches","create_image_summary") #여기서 에러 발생

# 테이블 + 요약 생성
#workflow.add_edge("get_page_numbers","crop_table")
workflow.add_edge("crop_table","create_table_summary_data_batches")
#workflow.add_edge("create_text_summary","create_table_summary_data_batches")
workflow.add_edge("create_table_summary_data_batches","create_table_summary")
workflow.add_edge("create_table_summary","create_table_markdown")

# 시작점 지정
workflow.set_entry_point("split_pdf")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

graph_png = app.get_graph(xray=True).draw_mermaid_png()

# 파일로 저장
with open("graph_output.png", "wb") as f:
    f.write(graph_png)
 
graph_config = RunnableConfig(recursion_limit=20, configurable={"thread_id": "Multi-RAG"})

inputs = GraphState(filepath=config["data"]["data_path"],batch_size=10)
output = app.invoke(inputs,config=graph_config)

save_graph_state(output, config["store"]["graph_state_path"])
state = load_graph_state(config["store"]["graph_state_path"])

hf_embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs={'device':'cuda'})

vectorstore = Chroma(
    collection_name="multi-modal-rag", embedding_function=hf_embeddings,
    persist_directory=config["store"]["vectorstore_directory"]
)

docstore = InMemoryStore()

add_documents_to_stores_and_save(
    vectorstore=vectorstore,
    docstore=docstore,
    text_summaries=state["texts_summary"],
    texts=state["texts"],
    table_summaries=state["tables_summary"],
    tables=state["tables"],
    image_summaries=state["images_summary"],
    images=state["images_base64"]
)
