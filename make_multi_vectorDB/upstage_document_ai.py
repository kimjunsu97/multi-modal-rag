import os
from graph_state.graph_state_manager import GraphState
import dotenv
import requests
import json

dotenv.load_dotenv()

class LayoutAnalyzer:
    def __init__(self, api_key):
        """
        LayoutAnalyzer 클래스의 생성자

        :param api_key: Upstage API 인증을 위한 API 키
        """
        self.api_key = api_key

    def _upstage_layout_analysis(self, input_file):
        """
        Upstage의 레이아웃 분석 API를 호출하여 문서 분석을 수행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        # API 요청 헤더 설정
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # API 요청 데이터 설정 (OCR 비활성화)
        data = {"ocr": False}

        # 분석할 PDF 파일 열기
        files = {"document": open(input_file, "rb")}

        # API 요청 보내기
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/layout-analysis",
            headers=headers,
            data=data,
            files=files,
        )

        # API 응답 처리 및 결과 저장
        if response.status_code == 200:
            # 분석 결과를 저장할 JSON 파일 경로 생성
            output_file = os.path.splitext(input_file)[0] + ".json"

            # 분석 결과를 JSON 파일로 저장
            with open(output_file, "w") as f:
                json.dump(response.json(), f, ensure_ascii=False)

            return output_file
        else:
            # API 요청이 실패한 경우 예외 발생
            raise ValueError(f"API 요청 실패. 상태 코드: {response.status_code}")

    def execute(self, input_file):
        """
        주어진 입력 파일에 대해 레이아웃 분석을 실행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        return self._upstage_layout_analysis(input_file)
    
def analyze_layout(state: GraphState):
    # 분할된 PDF 파일 목록을 가져옵니다.
    split_files = state["split_filepaths"]

    # LayoutAnalyzer 객체를 생성합니다. API 키는 환경 변수에서 가져옵니다.
    analyzer = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))

    # 분석된 파일들의 경로를 저장할 리스트를 초기화합니다.
    analyzed_files = []

    # 각 분할된 PDF 파일에 대해 레이아웃 분석을 수행합니다.
    for file in split_files:
        # 레이아웃 분석을 실행하고 결과 파일 경로를 리스트에 추가합니다.
        analyzed_files.append(analyzer.execute(file))

    # 분석된 파일 경로들을 정렬하여 새로운 GraphState 객체를 생성하고 반환합니다.
    # 정렬은 파일들의 순서를 유지하기 위해 수행됩니다.
    #state["analyzed_files"] = sorted(analyzed_files)
    #return state
    return GraphState(analyzed_files=sorted(analyzed_files))

# 임시용
# def analyze_layout(state: GraphState):
#     analyzed_files = ['./data/RAFT_0000_0000.json',
#   './data/RAFT_0001_0001.json',
#   './data/RAFT_0002_0002.json',
#   './data/RAFT_0003_0003.json',
#   './data/RAFT_0004_0004.json',
#   './data/RAFT_0005_0005.json',
#   './data/RAFT_0006_0006.json',
#   './data/RAFT_0007_0007.json',
#   './data/RAFT_0008_0008.json',
#   './data/RAFT_0009_0009.json',
#   './data/RAFT_0010_0010.json']
#     state["analyze_layout"] = sorted(analyzed_files)
#     return state