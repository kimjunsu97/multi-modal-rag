from graph_state.graph_state_manager import initialize_graph_state, save_graph_state, load_graph_state
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

print("--> Initializing a graph state")
state = initialize_graph_state(filepath="./data/RAFT.pdf", batch_size=10)

print("--> Splitting a pdf file")
state = split_pdf(state)

print("--> Analyzing a pdf layout")
state = analyze_layout(state)

print("--> Extracting a page metadata")
state = extract_page_metadata(state)

print("--> Extracting a page elements")
state = extract_page_elements(state)

print("--> Extracting a page tag element")
state = extract_tag_elements_per_page(state)

print("--> Extracting page numbers")
state = page_numbers(state)

print("--> Cropping images")
state = crop_image(state)

print("--> Generating base64 images")
state = generate_base64_image(state)

print("--> Cropping tables")
state = crop_table(state)

print("--> Extracting page texts")
state = extract_page_text(state)

print("--> Creating text summaries")
state = create_text_summary(state)

print("--> Creating image summary batches")
state = create_image_summary_data_batches(state)

print("--> Creating table summary batches")
state = create_table_summary_data_batches(state)

print("--> Creating image summaries")
state = create_image_summary(state)

print("--> Creating table summaries")
state = create_table_summary(state)

print("--> Creating table markdown")
state = create_table_markdown(state)

print("--> Saving graph_state")
save_graph_state(state, './store/graph_state/RAFT.pkl')

print("--> Loading graph_state")
state = load_graph_state('./store/graph_state/RAFT.pkl')

hf_embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs={'device':'cuda'})

vectorstore = Chroma(
    collection_name="sample-rag-multi-modal", embedding_function=hf_embeddings,
    persist_directory="./store/vectorstore/multi_modal_data"
)

docstore = InMemoryStore()

add_documents_to_stores_and_save(
    vectorstore=vectorstore,
    docstore=docstore,
    text_summaries=state["text_summary"],
    texts=state["texts"],
    table_summaries=state["table_summary"],
    tables=state["tables"],
    image_summaries=state["image_summary"],
    images=state["images_base64"]
)



