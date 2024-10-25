from graph_state.graph_state_manager import GraphState
def extract_page_text(state: GraphState):
    # 상태 객체에서 페이지 번호 목록을 가져옵니다.
    page_numbers = state["page_numbers"]

    # 추출된 텍스트를 저장할 딕셔너리를 초기화합니다.
    extracted_texts = dict()

    # 각 페이지 번호에 대해 반복합니다.
    for page_num in page_numbers:
        # 현재 페이지의 텍스트를 저장할 빈 문자열을 초기화합니다.
        extracted_texts[page_num] = ""

        # 현재 페이지의 모든 텍스트 요소에 대해 반복합니다.
        for element in state["page_elements"][page_num]["text_elements"]:
            # 각 텍스트 요소의 내용을 현재 페이지의 텍스트에 추가합니다.
            extracted_texts[page_num] += element["text"]

    # 추출된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    #state["texts"] = extracted_texts
    #return state
    return GraphState(texts=extracted_texts)

#--------------------------------------------------------------------------------------------#

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.documents import Document

# 요약을 위한 프롬프트 템플릿을 정의합니다.
prompt = PromptTemplate.from_template(
    """Please summarize the sentence according to the following REQUEST.
    
REQUEST:
1. Summarize the main points in bullet points.
2. Write the summary in same language as the context.
3. DO NOT translate any technical terms.
4. DO NOT include any unnecessary information.
5. Summary must include important entities, numerical values.

CONTEXT:
{context}

SUMMARY:"
"""
)

# ChatOpenAI 모델의 또 다른 인스턴스를 생성합니다. (이전 인스턴스와 동일한 설정)
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    #model_name="gpt-4o",
    temperature=0,
)

# 문서 요약을 위한 체인을 생성합니다.
# 이 체인은 여러 문서를 입력받아 하나의 요약된 텍스트로 결합합니다.
text_summary_chain = create_stuff_documents_chain(llm, prompt)

def create_text_summary(state: GraphState):
    # state에서 텍스트 데이터를 가져옵니다.
    texts = state["texts"]
    # 요약된 텍스트를 저장할 딕셔너리를 초기화합니다.
    text_summary = dict()

    # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])
    # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
    inputs = [
        {"context": [Document(page_content=text)]} for page_num, text in sorted_texts
    ]
    # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    summaries = text_summary_chain.batch(inputs)
    # 생성된 요약을 페이지 번호와 함께 딕셔너리에 저장합니다.
    for page_num, summary in enumerate(summaries):
        text_summary[page_num] = summary
    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    #state["texts_summary"] = text_summary
    #return state
    #print("text_summary: ",text_summary)
    return GraphState(texts_summary=text_summary)