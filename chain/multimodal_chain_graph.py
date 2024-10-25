from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from make_multi_vectorDB.image_utils import split_image_text_types, img_prompt_func
import dotenv
from graph_state.graph_state_chain import GraphState

dotenv.load_dotenv()
def multi_modal_rag_chain(retriever):
    """
    멀티모달 RAG 체인
    """

    # 멀티모달 LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=2048)

    # RAG 파이프라인
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

def multi_modal_rag_chain_graph(state:GraphState):
    """
    멀티모달 RAG 체인
    """

    # 멀티모달 LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=2048)
    context_runnable = RunnableLambda(lambda x: state["context"])

    # RAG 파이프라인
    chain = (
        {
            "context": context_runnable | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return GraphState(answer=chain.invoke(state["query"]))