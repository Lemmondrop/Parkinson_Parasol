# 파라솔(Parasol)
### (KDT Training Final Project)
---
<p align="center">
  <img width="300" height="600" alt="App logo" src="https://github.com/user-attachments/assets/96badb1a-0213-40f4-9ae9-52af149d9266" />
</p>

<p align="center">
  소중한 동행을 더 오래 이어주는 파라솔
</p>
<p align="center">
  <strong>Multi-Modal Digital Biomarker 기반 Parkinson's Disease(파킨슨병)의 초기 감별 AI 솔루션</strong>
</p>

<h2>🍀 Our Team</h2>

<table align="center">
  <tr>
    <th style="padding: 10px; font-size: 16px;">💡 이성현</th>
    <th style="padding: 10px; font-size: 16px;">💡 이은지</th>
    <th style="padding: 10px; font-size: 16px;">📊 신우철</th>
    <th style="padding: 10px; font-size: 16px;">📱 윤현수</th>
  </tr>
  <tr>
    <td align="center" style="padding: 10px; font-size: 14px;"><strong>챗봇 구현 &<br> 솔루션 기획</strong></td>
    <td align="center" style="padding: 10px; font-size: 14px;"><strong>UI&UX 디자인 <br>& 솔루션 기획</strong></td>
    <td align="center" style="padding: 10px; font-size: 14px;"><strong>AI 모델링 & <br>데이터 분석</strong></td>
    <td align="center" style="padding: 10px; font-size: 14px;"><strong>앱 개발 & <br>AWS Architect</strong></td>
  </tr>
</table>


## ⚙️ 기술 스택

<table>
  <tr>
    <th align="left">개발 환경</th>
    <td>
      <img src="https://img.shields.io/badge/Python-3.9.7-3776AB?logo=python&logoColor=white"/>
      <img src="https://img.shields.io/badge/Flutter-3.32.8-02569B?logo=flutter"/>
      <img src="https://img.shields.io/badge/Dart-3.8.1-0175C2?logo=dart"/>
      <img src="https://img.shields.io/badge/Android_Studio-Narwhal_3-3DDC84?logo=androidstudio"/>
    </td>
  </tr>
  <tr>
    <th align="left">백엔드</th>
    <td>
      <img src="https://img.shields.io/badge/python--multipart-0.0.9-4a4a4a"/>
      <img src="https://img.shields.io/badge/Pydantic-2.7.4-005571?logo=pydantic"/>
    </td>
  </tr>
  <tr>
    <th align="left">Data Preprocessing / ML</th>
    <td>
      <!-- Server side (requirements.txt) -->
      <img src="https://img.shields.io/badge/Pandas-2.2.2-150458?logo=pandas&logoColor=white"/>
      <img src="https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white"/>
      <img src="https://img.shields.io/badge/SciPy-1.13.1-8CAAE6?logo=scipy&logoColor=white"/>
      <img src="https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?logo=scikitlearn"/>
      <img src="https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c?logo=pytorch&logoColor=white"/>
      <img src="https://img.shields.io/badge/librosa-0.10.2.post1-1A1A1A"/>
      <img src="https://img.shields.io/badge/SoundFile-0.12.1-1A1A1A"/>
      <!-- Lambda (PROJECT_REQUIREMENTS.md) -->
      <img src="https://img.shields.io/badge/PyTorch(Lambda)-2.1.0-ee4c2c?logo=pytorch&logoColor=white"/>
      <img src="https://img.shields.io/badge/torchaudio-2.1.0-1A1A1A"/>
      <img src="https://img.shields.io/badge/librosa(Lambda)-0.10.1-1A1A1A"/>
      <img src="https://img.shields.io/badge/SoundFile(Lambda)-1.0.0-1A1A1A"/>
      <img src="https://img.shields.io/badge/scikit--learn(Lambda)-1.3.0-F7931E?logo=scikitlearn"/>
    </td>
  </tr>
  <tr>
    <th align="left">클라우드 / DB</th>
    <td>
      <img src="https://img.shields.io/badge/Firebase_Admin-6.5.0-FFCA28?logo=firebase"/>
      <img src="https://img.shields.io/badge/boto3-%3E%3D1.34.0-232F3E?logo=awslambda&logoColor=white"/>
      <img src="https://img.shields.io/badge/botocore-%3E%3D1.34.0-232F3E?logo=amazonaws&logoColor=white"/>
      <img src="https://img.shields.io/badge/Requests-%3E%3D2.31.0-4A4A4A"/>
    </td>
  </tr>
  <tr>
  <th align="left">컴퓨터 비전</th>
    <td>
      <img src="https://img.shields.io/badge/OpenCV(headless)-4.9.0.80-5C3EE8?logo=opencv&logoColor=white"/>
      <img src="https://img.shields.io/badge/MediaPipe-0.10.7-1A73E8?logo=google&logoColor=white"/>
      <!-- Lambda alt versions -->
      <img src="https://img.shields.io/badge/OpenCV(headless)(Lambda)-4.8.1.78-5C3EE8?logo=opencv&logoColor=white"/>
      <img src="https://img.shields.io/badge/Pillow-%3E%3D10.0.1-1A1A1A"/>
    </td>
  </tr>
</table>

## ✨ 프로젝트 주요 기능

### 🔍 주요 바이오마커
> 특발성 파킨슨병(PD), 비정형 파킨슨병(PSP, MSA), 정상(HC)을 구분할 수 있는 마커 입니다.<br>
> 시선 추적(PSP), 손가락 부딪치기(PD), 소리내기(MSA) 기준으로 탐색할 수 있어요.
<div align="center">

<table>
  <tr>
    <th align="center">시선 추적</th>
    <th align="center">손가락 부딪치기</th>
    <th align="center">소리내기</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/53fb25bf-cd5c-4f9b-9227-1fe31160ecc9" width="200"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/30e747c4-f72d-4be8-b7ef-9aec7a950128" width="200"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ee3ddd03-e118-434d-997f-890023787297" width="200"/>
    </td>
  </tr>
</table>

</div>

### 🧪 결과 보고서 & 아파닥
> 모든 측정 결과를 고려한 결과 보고서를 제공 받고<br>
> RAG(Retrieval-Augmented Generation)기반 파킨슨병(비정형 파킨슨병) 특화 챗봇과 함께 해보세요!
<div align="center">

<table>
  <tr>
    <th align="center">최종 위험도 보고서</th>
    <th align="center">아파닥</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2f0aba69-e079-4607-9970-8fb5d26cfdee" width="200"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8f96d860-7ba2-4e8e-8db3-a09397b0dd8d" width="200"/>
    </td>
  </tr>
</table>

</div>

## 🔧 아키텍처 다이어그램

<p align="center">
  <img src="https://github.com/user-attachments/assets/0a12bfff-dec4-4a52-bad2-9afc72c52353" alt="architecture-diagram" width="700"/>
</p>

> 전체 시스템은 **3-Tier 구조 기반의 경량 아키텍처**입니다.  
 - 빠른 응답성과 구조 단순화를 위해 3-Tier를 개조하여 경량화했습니다.  
 - 별도의 Service Layer 없이, API 내부에서 모든 로직을 직접 처리하도록 구성했습니다.

> **Firebase**를 데이터 저장소로, **FastAPI**를 중심으로 검색 / 분류 / 추천 기능을 제공합니다.  
 - 제품 데이터는 Firebase에 상세 필드를 모두 포함한 형태로 업로드되어 있습니다.  
 - 데이터베이스는 복잡한 관계형 구조를 지양하고, 단순성과 명확성을 우선했습니다.

> 앱은 **Flutter + Dart**로 개발되어 사용자 인터페이스를 담당합니다.

## 🛠️ My Work

### Search 로직 구현
- `rapidfuzz`를 활용하여 **오타/유사어 검색** 기능을 구현했습니다.
- 예: `롯데 → lotte`, `비비고 → bibigo` 등 한/영 브랜드 자동 매핑으로 **검색 편의성 강화**
- 유사도 기준이 높아 검색 실패 시 **조건을 자동 완화하는 Fallback 재검색 기능**을 추가했습니다.
- `자모 유사 검색` 로직을 통해 `"콜라 → 코카콜라"`, `"비비드" ↔ "vividkitchen"` 등도 매칭 가능하도록 설계했습니다.

### 제품 추천 기능
- **TF-IDF 기반 가중치 추천 알고리즘**을 직접 구현하였습니다.
- 원재료 성분 중 `감미료`, `주의 성분`의 포함 여부를 기반으로 가중치를 부여해 추천 점수를 계산합니다.
- 원재료 표 상에서 위험 성분의 위치가 **상위**일 경우 감점, **하위**일 경우 가점을 부여하여 보다 안전한 제품을 추천합니다.
