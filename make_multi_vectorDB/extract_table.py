from graph_state.graph_state_manager import GraphState
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal
from langchain_core.runnables import chain

def create_table_summary_data_batches(state: GraphState):
    # 테이블 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["page_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        text = state["texts_summary"][page_num]
        # 해당 페이지의 모든 테이블 요소에 대해 반복
        for image_element in state["page_elements"][page_num]["table_elements"]:
            # 테이블 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 테이블 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "table": state["tables"][image_id],  # 테이블 데이터
                    "text": text,  # 관련 텍스트 요약
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 테이블 ID
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    #state["table_summary_data_batches"] = data_batches
    #return state
    return GraphState(table_summary_data_batches=data_batches)

@chain
def extract_table_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o",  # 모델명
    )

    system_prompt = "You are an expert in extracting useful information from TABLE. With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["table"]
        user_prompt_template = f"""Here is the context related to the image of table: {context}
        
###

Output Format:

<table>
<title>
<table_summary>
<key_entities> 
<data_insights>
</table>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer

def create_table_summary(state: GraphState):
    # 테이블 요약 추출
    table_summaries = extract_table_summary.invoke(
        state["table_summary_data_batches"],
    )

    # 테이블 요약 결과를 저장할 딕셔너리 초기화
    table_summary_output = dict()

    # 각 데이터 배치와 테이블 요약을 순회하며 처리
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 테이블 요약 저장
        table_summary_output[data_batch["id"]] = table_summary

    # 테이블 요약 결과를 포함한 새로운 GraphState 객체 반환
    #state["table_summary"] = table_summary_output
    #return state
    return GraphState(tables_summary=table_summary_output)

@chain
def table_markdown_extractor(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = "You are an expert in converting image of the TABLE into markdown format. Be sure to include all the information in the table. DO NOT narrate, just answer in markdown format."

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        image_path = data_batch["table"]
        user_prompt_template = f"""DO NOT wrap your answer in ```markdown``` or any XML tags.
        
###

Output Format:

<table_markdown>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer

def create_table_markdown(state: GraphState):
    # table_markdown_extractor를 사용하여 테이블 마크다운 생성
    # state["table_summary_data_batches"]에 저장된 테이블 데이터를 사용
    table_markdowns = table_markdown_extractor.invoke(
        state["table_summary_data_batches"],
    )

    # 결과를 저장할 딕셔너리 초기화
    table_markdown_output = dict()

    # 각 데이터 배치와 생성된 테이블 마크다운을 매칭하여 저장
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_markdowns
    ):
        # 데이터 배치의 id를 키로 사용하여 테이블 마크다운 저장
        table_markdown_output[data_batch["id"]] = table_summary

    # 새로운 GraphState 객체 반환, table_markdown 키에 결과 저장
    #state["table_markdown"]=table_markdown_output
    #return state
    return GraphState(table_markdown=table_markdown_output)
