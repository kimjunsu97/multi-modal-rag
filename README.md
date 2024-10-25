# Multi-Modal-RAG
이 프로젝트는 Multi-Vectorstore를 활용하여 텍스트, 표, 이미지 데이터를 결합한 멀티모달 RAG 시스템을 구축하고, 이를 통해 LLM이 다양한 형식의 데이터를 기반으로 응답을 생성하도록 하는 것을 목표로 합니다. 본 프로젝트에서는 여러 소스에서 수집한 데이터를 바탕으로, LLM이 더욱 정교하고 풍부한 답변을 제공할 수 있는 멀티모달 문서 검색 및 생성 시스템을 구현합니다. 이를 통해 다양한 실제 애플리케이션에 적용할 수 있는 최적의 솔루션을 도출하는 데 목적이 있습니다.
## env 설정

```
$git clone https://github.com/kimjunsu97/multi-modal-rag.git
$cd multi-modal-rag
$pip install -r requirements.txt
```
## 데이터 준비 방법
./data 폴더안에 .pdf 파일을 넣고 config.yaml 파일에 경로 지정

## Multi-Vectorstore 방법
```
$python make_multi_vectordb.py
or
$python make_multi_vectordb_graph.py
# 구현 방식은 다르나 결과는 동일
```

## Multi-Modal LLM RAG 실행 방법
```
$python multi_modla_rag_basic.py
```


