from graph_state.graph_state_manager import GraphState
from langchain_openai import ChatOpenAI

from langchain_teddynote.models import MultiModal
from langchain_core.runnables import chain

def create_image_summary_data_batches(state: GraphState):
    # 이미지 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["page_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        text = state["texts_summary"][page_num]
        # 해당 페이지의 모든 이미지 요소에 대해 반복
        for image_element in state["page_elements"][page_num]["image_elements"]:
            # 이미지 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 이미지 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "image": state["images"][image_id],  # 이미지 파일 경로
                    "text": text,  # 관련 텍스트 요약
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 이미지 ID
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    #state["image_summary_data_batches"] = data_batches
    #return state
    return GraphState(image_summary_data_batches=data_batches)


@chain
def extract_image_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting useful information from IMAGE.
    With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["image"]
        user_prompt_template = f"""Here is the context related to the image: {context}
        
###

Output Format:

<image>
<title>
<summary>
<entities> 
</image>

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

def create_image_summary(state: GraphState):
    # 이미지 요약 추출
    # extract_image_summary 함수를 호출하여 이미지 요약 생성
    image_summaries = extract_image_summary.invoke(
        state["image_summary_data_batches"],
    )

    # 이미지 요약 결과를 저장할 딕셔너리 초기화
    image_summary_output = dict()

    # 각 데이터 배치와 이미지 요약을 순회하며 처리
    for data_batch, image_summary in zip(
        state["image_summary_data_batches"], image_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 이미지 요약 저장
        image_summary_output[data_batch["id"]] = image_summary

    # 이미지 요약 결과를 포함한 새로운 GraphState 객체 반환
    #state["image_summary"]=image_summary_output
    #return state
    return GraphState(images_summary=image_summary_output)