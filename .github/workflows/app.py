import io
import os
import re
import time
import requests
import pandas as pd
import gradio as gr

# =====[ 깃허브 RAW CSV 경로 설정 ]=====
OWNER  = "kimjongwook-ui"   # <- 네 깃허브 아이디
REPO   = "slotchatbot"      # <- 리포 이름
BRANCH = "data"             # <- 기본 브랜치 (다르면 변경)
CSV_PATH_IN_REPO = "game_info.csv"
CSV_URL = f"https://raw.githubusercontent.com/kimjongwook-ui/slotchatbot/main/game_info.csv"

# 캐시(10분 TTL)
_df_cache = None
_last_loaded = 0.0
TTL_SEC = 600

# =====[ 유틸: DF 로드 ]=====
def load_df(force: bool = False) -> pd.DataFrame:
    global _df_cache, _last_loaded
    now = time.time()
    if (not force) and _df_cache is not None and (now - _last_loaded) < TTL_SEC:
        return _df_cache
    r = requests.get(CSV_URL, timeout=20)
    r.raise_for_status()
    _df_cache = pd.read_csv(io.StringIO(r.text))
    _last_loaded = now
    return _df_cache

# =====[ 슬롯챗봇버전1: 컬럼 매핑/정규화 ]=====
def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 동일 이름 충돌 방지용: 원본 컬럼명 보존
    return df

# 후보 컬럼 자동 탐색(여러 데이터셋에 맞게 느슨하게)
def find_col(df: pd.DataFrame, candidates) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        # 완전일치 우선
        if key in cols:
            return cols[key]
    # 부분일치도 허용
    for c in df.columns:
        lc = c.lower()
        for key in candidates:
            if key.lower() in lc:
                return c
    return None

# 표준 컬럼 이름들 추정
def detect_schema(df: pd.DataFrame) -> dict:
    name_col = find_col(df, ["title", "name", "게임명", "slot", "game"])
    provider_col = find_col(df, ["provider", "studio", "제작사", "게임사"])
    theme_col = find_col(df, ["theme", "테마"])
    features_col = find_col(df, ["features", "feature", "기능"])
    overall_rtp_col = find_col(df, ["rtp", "overall_rtp", "total_rtp", "전체 rtp"])
    base_rtp_col = find_col(df, ["base_rtp", "base rtp", "베이스 rtp"])
    free_rtp_col = find_col(df, ["free_rtp", "free rtp", "프리 rtp", "freespin rtp"])
    volatility_col = find_col(df, ["volatility", "variance", "편차", "볼라"])
    hit_rate_col = find_col(df, ["hit_rate", "hit rate", "히트", "히트율"])
    lines_col = find_col(df, ["lines", "line", "라인"])
    ways_col = find_col(df, ["ways", "way", "웨이"])
    size_col = find_col(df, ["size", "grid", "릴행", "릴x행", "릴수x행수"])
    reels_col = find_col(df, ["reels", "reel", "릴", "columns"])
    rows_col = find_col(df, ["rows", "row", "행"])

    return {
        "name": name_col,
        "provider": provider_col,
        "theme": theme_col,
        "features": features_col,
        "overall_rtp": overall_rtp_col,
        "base_rtp": base_rtp_col,
        "free_rtp": free_rtp_col,
        "volatility": volatility_col,
        "hit_rate": hit_rate_col,
        "lines": lines_col,
        "ways": ways_col,
        "size": size_col,
        "reels": reels_col,
        "rows": rows_col,
    }

# =====[ 슬롯챗봇버전1: 쿼리 파서 ]=====
NUM = r"(?P<num>\d+(?:\.\d+)?)"
RANGE = r"(?P<n1>\d+(?:\.\d+)?)[\s]*(?:~|-|to|~|–|—|까지|~)\s*(?P<n2>\d+(?:\.\d+)?)"
SIZE = r"(?P<r>\d+)\s*[xX×]\s*(?P<c>\d+)"

# 한/영 키워드 동의어(컬럼 지시어)
FIELD_ALIASES = {
    "rtp_overall": [r"\boverall\s*rtp\b", r"\btotal\s*rtp\b", r"\brtp\b", r"전체\s*rtp"],
    "rtp_base": [r"베이스\s*rtp", r"\bbase\s*rtp\b"],
    "rtp_free": [r"프리\s*rtp", r"free\s*rtp", r"freespin\s*rtp"],
    "volatility": [r"편차", r"볼라", r"\bvolatility\b", r"\bvariance\b"],
    "hit_rate": [r"히트(율)?", r"hit\s*rate"],
    "lines": [r"라인", r"\blines?\b"],
    "ways": [r"웨이", r"\bways?\b"],
    "size": [r"\bsize\b", r"사이즈", r"그리드"],
}

# 비교 연산 한국어 → 기호
CMP_WORD = {"이상": ">=", "이하": "<=", "초과": ">", "미만": "<", "같음": "=="}

# Land&Win 류 동의어 → 키워드로 취급(AND 매칭에 참여)
LANDWIN_ALIASES = [
    "land&win", "land and win", "랜드앤윈", "랜앤윈", "hold&win", "hold and win",
    "코인", "코인피처", "coin", "coin hold", "coin feature"
]

def match_any(patterns, text):
    for p in patterns:
        if re.search(p, text, flags=re.I):
            return True
    return False

def extract_numeric_clause(field_key: str, text: str):
    """
    (예)
    - rtp 95~97
    - 편차 11 이상
    - lines >= 25
    - rtp 95 (단일 숫자 ±0.5)
    """
    clauses = []

    # 1) 범위(95~97)
    range_pat = rf"({ '|'.join(FIELD_ALIASES[field_key]) })\s*(?:{RANGE})"
    for m in re.finditer(range_pat, text, flags=re.I):
        n1 = float(m.group("n1")); n2 = float(m.group("n2"))
        clauses.append(("between", (min(n1, n2), max(n1, n2))))
    if clauses:
        return clauses

    # 2) 비교 연산(이상/이하/초과/미만, 기호)
    cmp_words = "|".join(CMP_WORD.keys())
    cmp_ops = r"[<>]=?|=="
    cmp_pat = rf"({ '|'.join(FIELD_ALIASES[field_key]) })\s*(?:({cmp_ops}|{cmp_words}))\s*{NUM}%?"
    for m in re.finditer(cmp_pat, text, flags=re.I):
        op = m.group(2)
        val = float(m.group("num"))
        if op in CMP_WORD:
            op = CMP_WORD[op]
        clauses.append((op, val))
    if clauses:
        return clauses

    # 3) 단일 숫자(±0.5)
    single_pat = rf"({ '|'.join(FIELD_ALIASES[field_key]) })\s*{NUM}%?"
    for m in re.finditer(single_pat, text, flags=re.I):
        val = float(m.group("num"))
        clauses.append(("between", (val - 0.5, val + 0.5)))
    return clauses

def parse_query(text: str) -> dict:
    q = text.strip()
    res = {
        "size": None,     # (reels, rows)
        "lines": [],      # list of (op,val) or ("between",(a,b))
        "ways": [],
        "rtp_overall": [],
        "rtp_base": [],
        "rtp_free": [],
        "volatility": [],
        "hit_rate": [],
        "keywords": [],   # AND 키워드(남는 단어들)
        "landwin": False, # 랜드앤윈 동의어 감지
    }
    if not q:
        return res

    # 사이즈 5x3
    m = re.search(SIZE, q, flags=re.I)
    if m:
        res["size"] = (int(m.group("r")), int(m.group("c")))

    # 수치 조건들
    for fk in ["rtp_overall", "rtp_base", "rtp_free", "volatility", "hit_rate", "lines", "ways"]:
        clauses = extract_numeric_clause(fk, q)
        if clauses:
            res[fk] = clauses

    # 랜드앤윈/홀앤윈 류 키워드 감지
    res["landwin"] = match_any([re.escape(w) for w in LANDWIN_ALIASES], q)

    # 남는 키워드(AND 매칭)
    # 숫자/연산/특수문자 제거 후 토큰화
    cleaned = re.sub(SIZE, " ", q, flags=re.I)
    cleaned = re.sub(r"\d+(\.\d+)?\s*(%|라인|웨이)?", " ", cleaned)
    cleaned = re.sub(r"[><=~\-:]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    toks = [t.strip().lower() for t in cleaned.split() if t.strip()]
    # 의미없는 토큰 제거
    stop = set(["이상","이하","초과","미만","부터","까지","사이","and","&","에서","between","to","the","a","of"])
    toks = [t for t in toks if t not in stop]
    # 이미 필드 지시어에 포함된 단어는 제외
    field_words = set()
    for arr in FIELD_ALIASES.values():
        for pat in arr:
            field_words.update(re.findall(r"[가-힣A-Za-z]+", pat.lower()))
    toks = [t for t in toks if t not in field_words]
    res["keywords"] = toks
    return res

# =====[ 슬롯챗봇버전1: 필터링 로직 ]=====
def between_mask(series: pd.Series, a: float, b: float):
    try:
        return (pd.to_numeric(series, errors="coerce") >= a) & (pd.to_numeric(series, errors="coerce") <= b)
    except Exception:
        return pd.Series([False]*len(series), index=series.index)

def cmp_mask(series: pd.Series, op: str, val: float):
    s = pd.to_numeric(series, errors="coerce")
    if op == ">=": return s >= val
    if op == "<=": return s <= val
    if op == ">":  return s >  val
    if op == "<":  return s <  val
    if op == "==": return s == val
    return pd.Series([True]*len(series), index=series.index)

def apply_numeric_filters(df: pd.DataFrame, col: str | None, clauses: list):
    if not clauses or not col or col not in df.columns:
        return pd.Series([True]*len(df), index=df.index)
    mask = pd.Series([True]*len(df), index=df.index)
    for op, val in clauses:
        if op == "between":
            a, b = val
            mask &= between_mask(df[col], a, b)
        else:
            mask &= cmp_mask(df[col], op, val)
    return mask

def filter_size(df: pd.DataFrame, schema: dict, size_tuple):
    if not size_tuple:
        return pd.Series([True]*len(df), index=df.index)
    r, c = size_tuple
    # reels/rows 우선
    if schema["reels"] and schema["rows"]:
        s1 = pd.to_numeric(df[schema["reels"]], errors="coerce") == r
        s2 = pd.to_numeric(df[schema["rows"]], errors="coerce") == c
        return s1 & s2
    # size 문자열 파싱(예: "5x3")
    if schema["size"]:
        def parse_size(x):
            if pd.isna(x): return False
            m = re.search(SIZE, str(x), flags=re.I)
            return bool(m and int(m.group("r")) == r and int(m.group("c")) == c)
        return df[schema["size"]].apply(parse_size)
    # 매칭 불가시 전체 True
    return pd.Series([True]*len(df), index=df.index)

def keyword_and_match(df: pd.DataFrame, schema: dict, toks: list[str], landwin_flag: bool):
    if not toks and not landwin_flag:
        return pd.Series([True]*len(df), index=df.index)

    text_cols = [c for c in [
        schema["name"], schema["provider"], schema["theme"], schema["features"]
    ] if c and c in df.columns]
    if not text_cols:
        # 텍스트 칼럼 추정 실패 시 전체 텍스트 합쳐 AND 매칭
        text_cols = [c for c in df.columns if df[c].dtype == "object"]

    # Land&Win은 토큰에 자동 추가
    if landwin_flag:
        toks = toks + ["hold", "win", "land", "랜드", "홀드"]  # 느슨한 매칭

    # 모든 토큰이 "어느 한 텍스트 칼럼에라도" 존재해야 함 (AND)
    mask = pd.Series([True]*len(df), index=df.index)
    for t in toks:
        tmask = pd.Series([False]*len(df), index=df.index)
        for c in text_cols:
            tmask = tmask | df[c].astype(str).str.contains(re.escape(t), case=False, na=False)
        mask = mask & tmask
    return mask

def format_result(df: pd.DataFrame, schema: dict, limit=20) -> str:
    # 보여줄 열 구성
    cols_pref = [schema["name"], schema["provider"], schema["overall_rtp"] or schema["base_rtp"] or schema["free_rtp"],
                 schema["volatility"], schema["hit_rate"],
                 schema["lines"] or schema["ways"],
                 schema["size"] or schema["reels"] or schema["rows"],
                 schema["theme"]]
    cols = [c for c in cols_pref if c and c in df.columns]
    if not cols:
        cols = list(df.columns)[:8]
    out = df.loc[:, cols].head(limit)
    try:
        return out.to_markdown(index=False)
    except Exception:
        return out.to_string(index=False)

def kanana_query(message: str, df: pd.DataFrame) -> str:
    if not message or not message.strip():
        return "검색어를 입력하세요. 예) RTP 95% 이상 & 편차 11 이상, 5x3, 라인 25 이상"
    schema = detect_schema(df)
    q = parse_query(message)

    # 숫자/사이즈 필터
    mask = pd.Series([True]*len(df), index=df.index)
    mask &= filter_size(df, schema, q["size"])
    mask &= apply_numeric_filters(df, schema["overall_rtp"] or schema["base_rtp"] or schema["free_rtp"], q["rtp_overall"])
    mask &= apply_numeric_filters(df, schema["base_rtp"], q["rtp_base"])
    mask &= apply_numeric_filters(df, schema["free_rtp"], q["rtp_free"])
    mask &= apply_numeric_filters(df, schema["volatility"], q["volatility"])
    mask &= apply_numeric_filters(df, schema["hit_rate"], q["hit_rate"])
    # 라인/웨이(둘 다 있으면 각각 적용 → 교집합)
    mask &= apply_numeric_filters(df, schema["lines"], q["lines"])
    mask &= apply_numeric_filters(df, schema["ways"], q["ways"])

    # 자유 키워드 AND 매칭
    mask &= keyword_and_match(df, schema, q["keywords"], q["landwin"])

    result = df[mask]
    if result.empty:
        return "검색 결과가 없습니다."
    return format_result(result, schema, limit=30)

# =====[ Gradio UI ]=====
with gr.Blocks(title="Kanana 슬롯 검색 — 슬롯챗봇버전1") as demo:
    gr.Markdown("### Kanana 슬롯 검색 — 슬롯챗봇버전1 (GitHub CSV 연동)")

    chat = gr.Chatbot(height=460)
    box = gr.Textbox(
        placeholder="예) RTP 95% 이상 & 편차 11 이상, 5x3, 라인 25 이상, 랜드앤윈, 카툰 테마",
        label="질문/조건"
    )
    reload_btn = gr.Button("데이터 새로고침")

    def respond(user_msg, history):
        try:
            df = load_df()
            ans = kanana_query(user_msg, df)
        except Exception as e:
            ans = f"에러: {e}"
        return "", history + [(user_msg, ans)]

    def reload_df():
        load_df(force=True)
        return "최신 CSV로 새로 불러왔습니다."

    box.submit(respond, [box, chat], [box, chat])
    reload_btn.click(lambda: (reload_df(),), None, None)

if __name__ == "__main__":

    demo.launch()
