# Warhammer Age of Sigmar AI 룰마스터

워해머 미니어처 게임 정보를 한곳에 모아, 플레이 중 규칙·팩션 정보를 빠르게 찾고 이해할 수 있도록 돕는 RAG 기반 AI 도우미입니다.

---

## 기획의도

- **워해머 에이지 오브 지그마** 관련 공식 문서(룰북, FAQ, 팩션, 스피어헤드 등)를 한곳에 모아
- 게임 플레이 중 **규칙 해석·설명**과 **팩션별 정보**를 자연어 질문으로 바로 조회
- 공식 PDF 기반 답변으로 **오해·할루시네이션을 줄이고**, 출처를 함께 제공

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **룰북 해석** | 코어 룰, FAQ, 배틀스크롤 등 규칙 문서를 벡터 검색 후 AI가 해석·요약 |
| **룰북 설명** | "돌격 이동 규칙이 뭐야?", "스피어헤드 규칙 알려줘" 등 질문에 문서 기반으로 답변 |
| **팩션별 정보** | 각 팩션 PDF를 인덱싱해, 특정 팩션 규칙·유닛·특수 규칙 질의 지원 |
| **출처 표시** | 답변 시 참고한 문서 제목·URL·유사도 거리를 함께 표시 |

---

## 프로젝트 구조

```
rag-test/
├── chunk.py      # Warhammer Community 다운로드 페이지 스크래핑 → rule/faction/spearhead 분류
├── build_db.py   # PDF 다운로드·텍스트 추출·청킹·임베딩 후 ChromaDB에 저장
├── app.py        # Streamlit 웹 앱 (채팅 UI, RAG 질의·답변)
├── chat.py       # CLI용 RAG 챗 (Gemini + rule_db)
├── read.py       # CLI용 벡터 검색 테스트 (rule/faction/spearhead)
├── .env          # API 키 설정 (GEMINI_API_KEY, FIRECRAWL_API_KEY)
└── my_warhammer_db/   # ChromaDB 영구 저장 경로 (실행 후 생성)
```

---

## 기술 스택

- **벡터 DB**: ChromaDB (cosine 유사도)
- **임베딩**: SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`)
- **LLM**: Google Gemini (`gemini-2.5-flash`)
- **스크래핑**: Firecrawl (다운로드 페이지), `pypdf` (PDF 텍스트 추출)
- **웹 UI**: Streamlit

---

## 설치 및 설정

### 1. 의존성 설치

```bash
cd rag-test
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install streamlit chromadb sentence-transformers google-genai python-dotenv firecrawl-py pypdf requests tqdm
```

### 2. 환경 변수

프로젝트 루트에 `.env` 파일을 만들고 다음 키를 설정합니다.

```env
GEMINI_API_KEY=your_gemini_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

- **GEMINI_API_KEY**: [Google AI Studio](https://aistudio.google.com/)에서 발급
- **FIRECRAWL_API_KEY**: [Firecrawl](https://firecrawl.dev/)에서 발급 (chunk.py에서 다운로드 페이지 스크래핑 시 사용)

### 3. DB 구축 (최초 1회)

1. **문서 목록 수집 및 벡터 DB 저장**  
   `chunk.py`는 다운로드 페이지를 스크래핑해 `rule_db`, `faction_db`, `spearhead_db` 리스트를 만들고, ChromaDB에 PDF를 청킹·임베딩해 저장합니다.

   ```bash
   python chunk.py
   ```

2. **이미 목록이 있을 때 DB만 재구축**  
   `build_db.py`는 `chunk.py`에서 정의된 `rule_db`, `faction_db`, `spearhead_db`가 같은 프로세스/파일에서 사용 가능해야 합니다.  
   (실제로는 `chunk.py`에서 리스트를 만든 뒤 `build_db.py`를 import하거나, `build_db.py`에서 해당 리스트를 정의해 사용하는 방식으로 맞춰야 합니다. 현재는 `chunk.py` 실행 시 스크래핑 + DB 저장까지 수행하는 구조로 보입니다.)

   DB만 따로 다시 만들고 싶다면 `build_db.py`에 `rule_db`, `faction_db`, `spearhead_db`를 채워 넣은 뒤:

   ```bash
   python build_db.py
   ```

---

## 실행 방법

### 웹 앱 (권장)

```bash
streamlit run app.py
```

브라우저에서 열리는 채팅 화면에서 규칙·팩션 관련 질문을 입력하면, RAG 기반으로 답변과 출처가 표시됩니다.

### CLI 검색 (벡터 검색만)

```bash
python read.py
```

`read.py` 내부의 `search_optimized_chunks(..., rule_collection)` 등 호출을 바꿔서 `rule_collection` / `faction_collection` / `spearhead_collection`에 대해 테스트할 수 있습니다.

### CLI 챗 (RAG + Gemini)

```bash
python chat.py
```

`chat.py` 끝부분의 `ask_warhammer_rule_gemini("...", rule_collection)` 인자를 바꿔서 질문 문구와 컬렉션(rule/faction/spearhead)을 바꿀 수 있습니다.

---

## 데이터 출처

- 규칙·팩션·스피어헤드 PDF 링크는 [Warhammer Community - Age of Sigmar Downloads](https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/) 페이지에서 수집됩니다.
- 상업적 재배포가 아닌 개인/로컬 활용 목적의 스크래핑·인덱싱임을 전제로 합니다.

---

## 라이선스 및 면책

- 이 프로젝트는 Warhammer/게임즈 워크숍의 공식 제품이 아니며, 규칙 해석은 참고용입니다.  
- 공식 토너먼트·공식 판단이 필요한 경우 반드시 공식 룰북과 GW 공지에 의존하세요.
