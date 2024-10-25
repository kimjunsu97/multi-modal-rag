from typing import TypedDict
import pickle

# GraphState 상태를 저장하는 용도로 사용합니다.
class GraphState(TypedDict):
    filepath: str  # path
    filetype: str  # pdf
    page_numbers: list[int]  # page numbers
    batch_size: int  # batch size
    split_filepaths: list[str]  # split files
    analyzed_files: list[str]  # analyzed files
    page_elements: dict[int, dict[str, list[dict]]]  # page elements
    page_metadata: dict[int, dict]  # page metadata
    page_summary: dict[int, str]  # page summary
    images: list[str]  # image paths
    images_base64: list[str] #base64 image
    image_summary_data_batches: list[dict]
    images_summary: list[str]  # image summary
    tables: list[str]  # table
    table_markdown: list[str]
    table_summary_data_batches: list[dict]
    tables_summary: dict[int, str]  # table summary
    texts: list[str]  # text
    texts_summary: list[str]  # text summary



# GraphState 초기화 함수
def initialize_graph_state(filepath: str, batch_size: int) -> GraphState:
    return GraphState(
        filepath=filepath,
        filetype="pdf",
        page_numbers=[],
        batch_size=batch_size,
        split_filepaths=[],
        analyzed_files=[], 
        page_elements={}, 
        page_metadata={}, 
        page_summary={}, 
        images=[], 
        images_base64=[], 
        images_summary=[], 
        tables=[], 
        tables_summary={}, 
        texts=[], 
        texts_summary=[]
    )


# GraphState를 파일로 저장하는 함수
def save_graph_state(state: GraphState, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

# GraphState를 파일에서 불러오는 함수
def load_graph_state(filepath: str) -> GraphState:
    with open(filepath, 'rb') as f:
        return pickle.load(f)