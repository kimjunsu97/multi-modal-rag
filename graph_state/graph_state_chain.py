from typing import TypedDict, Any

# GraphState 상태를 저장하는 용도로 사용합니다.
class GraphState(TypedDict):
    query: str
    docstore_path: str  # docstore_path
    vectorstore_directory: str  # vectorestore_directory
    retriever: Any # retriever
    context: Any # context
    answer: str # answer
    