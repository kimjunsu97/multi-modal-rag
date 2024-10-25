import re
import os
import json
from graph_state.graph_state_manager import GraphState

def extract_start_end_page(filename):
    """
    파일 이름에서 시작 페이지와 끝 페이지 번호를 추출하는 함수입니다.

    :param filename: 분석할 파일의 이름
    :return: 시작 페이지 번호와 끝 페이지 번호를 튜플로 반환
    """
    # 파일 경로에서 파일 이름만 추출
    file_name = os.path.basename(filename)
    # 파일 이름을 '_' 기준으로 분리
    file_name_parts = file_name.split("_")

    if len(file_name_parts) >= 3:
        # 파일 이름의 뒤에서 두 번째 부분에서 숫자를 추출하여 시작 페이지로 설정
        start_page = int(re.findall(r"(\d+)", file_name_parts[-2])[0])
        # 파일 이름의 마지막 부분에서 숫자를 추출하여 끝 페이지로 설정
        end_page = int(re.findall(r"(\d+)", file_name_parts[-1])[0])
    else:
        # 파일 이름 형식이 예상과 다를 경우 기본값 설정
        start_page, end_page = 0, 0

    return start_page, end_page


def extract_page_metadata(state: GraphState):
    """
    분석된 JSON 파일들에서 페이지 메타데이터를 추출하는 함수입니다.

    :param state: 현재의 GraphState 객체
    :return: 페이지 메타데이터가 추가된 새로운 GraphState 객체
    """
    # 분석된 JSON 파일 목록 가져오기
    json_files = state["analyzed_files"]

    # 페이지 메타데이터를 저장할 딕셔너리 초기화
    page_metadata = dict()

    for json_file in json_files:
        # JSON 파일 열기 및 데이터 로드
        with open(json_file, "r") as f:
            data = json.load(f)

        # 파일명에서 시작 페이지 번호 추출
        start_page, _ = extract_start_end_page(json_file)

        # JSON 데이터에서 각 페이지의 메타데이터 추출
        for element in data["metadata"]["pages"]:
            # 원본 페이지 번호
            original_page = int(element["page"])
            # 상대적 페이지 번호 계산 (전체 문서 기준)
            relative_page = start_page + original_page - 1

            # 페이지 크기 정보 추출
            metadata = {
                "size": [
                    int(element["width"]),
                    int(element["height"]),
                ],
            }
            # 상대적 페이지 번호를 키로 하여 메타데이터 저장
            page_metadata[relative_page] = metadata
    
    # 추출된 페이지 메타데이터로 새로운 GraphState 객체 생성 및 반환
    #state["page_metadata"] = page_metadata
    #return state
    return GraphState(page_metadata=page_metadata)

def extract_page_elements(state: GraphState):
    # 분석된 JSON 파일 목록을 가져옵니다.
    json_files = state["analyzed_files"]

    # 페이지별 요소를 저장할 딕셔너리를 초기화합니다.
    page_elements = dict()

    # 전체 문서에서 고유한 요소 ID를 부여하기 위한 카운터입니다.
    element_id = 0

    # 각 JSON 파일을 순회하며 처리합니다.
    for json_file in json_files:
        # 파일명에서 시작 페이지 번호를 추출합니다.
        start_page, _ = extract_start_end_page(json_file)

        # JSON 파일을 열어 데이터를 로드합니다.
        with open(json_file, "r") as f:
            data = json.load(f)

        # JSON 데이터의 각 요소를 처리합니다.
        for element in data["elements"]:
            # 원본 페이지 번호를 정수로 변환합니다.
            original_page = int(element["page"])
            # 전체 문서 기준의 상대적 페이지 번호를 계산합니다.
            relative_page = start_page + original_page - 1

            # 해당 페이지의 요소 리스트가 없으면 새로 생성합니다.
            if relative_page not in page_elements:
                page_elements[relative_page] = []

            # 요소에 고유 ID를 부여합니다.
            element["id"] = element_id
            element_id += 1

            # 요소의 페이지 번호를 상대적 페이지 번호로 업데이트합니다.
            element["page"] = relative_page
            # 요소를 해당 페이지의 리스트에 추가합니다.
            page_elements[relative_page].append(element)

    # 추출된 페이지별 요소 정보로 새로운 GraphState 객체를 생성하여 반환합니다.
    #state["page_elements"] = page_elements
    #return state
    return GraphState(page_elements=page_elements)

    #각 페이지의 태그를 분해합니다.
def extract_tag_elements_per_page(state: GraphState):
    # GraphState 객체에서 페이지 요소들을 가져옵니다.
    page_elements = state["page_elements"]

    # 파싱된 페이지 요소들을 저장할 새로운 딕셔너리를 생성합니다.
    parsed_page_elements = dict()

    # 각 페이지와 해당 페이지의 요소들을 순회합니다.
    for key, page_element in page_elements.items():
        # 이미지, 테이블, 텍스트 요소들을 저장할 리스트를 초기화합니다.
        image_elements = []
        table_elements = []
        text_elements = []

        # 페이지의 각 요소를 순회하며 카테고리별로 분류합니다.
        for element in page_element:
            if element["category"] == "figure":
                # 이미지 요소인 경우 image_elements 리스트에 추가합니다.
                image_elements.append(element)
            elif element["category"] == "table":
                # 테이블 요소인 경우 table_elements 리스트에 추가합니다.
                table_elements.append(element)
            else:
                # 그 외의 요소는 모두 텍스트 요소로 간주하여 text_elements 리스트에 추가합니다.
                text_elements.append(element)

        # 분류된 요소들을 페이지 키와 함께 새로운 딕셔너리에 저장합니다.
        parsed_page_elements[key] = {
            "image_elements": image_elements,
            "table_elements": table_elements,
            "text_elements": text_elements,
            "elements": page_element,  # 원본 페이지 요소도 함께 저장합니다.
        }

    # 파싱된 페이지 요소들을 포함한 새로운 GraphState 객체를 반환합니다.
    #state["page_elements"] = parsed_page_elements
    #return state
    return GraphState(page_elements=parsed_page_elements)

# 페이지 번호를 추출
def page_numbers(state: GraphState):
    #state["page_numbers"] = list(state["page_elements"].keys())
    #return state
    return GraphState(page_numbers=list(state["page_elements"].keys()))
