
# app.py — Kanana 슬롯 검색 (HF Spaces, Gradio 4.44.0)
# 버전: 슬롯서치완성1.3-external (외부 슬롯 rgs/sng + 2nd CSV 지원)
import os, io, re, time, requests
import pandas as pd
import gradio as gr

# ====== 고정 설정 ======
OWNER, REPO, BRANCH = "kimjongwook-ui", "slotchatbot", "data"
CSV_PATH_MAIN      = "game_info.csv"             # 내부 슬롯(기존)
CSV_PATH_EXTERNAL  = "external_game_info.csv"    # 외부 슬롯(rgs/sng)
CSV_URL_MAIN      = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{CSV_PATH_MAIN}"
CSV_URL_EXTERNAL  = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{CSV_PATH_EXTERNAL}"
TTL_SEC = 600  # CSV 캐시 10분

# (선택) 로컬 우선 테스트 경로. 지정시 로컬 파일 사용 후, 없으면 GitHub URL 사용.
CSV_LOCAL_MAIN     = os.getenv("CSV_LOCAL_MAIN")     # 예: /data/game_info.csv
CSV_LOCAL_EXTERNAL = os.getenv("CSV_LOCAL_EXTERNAL") # 예: /data/external_game_info.csv

# ====== 내부 캐시 ======
_df_cache_main = None
_df_cache_ext  = None
_last_loaded_main = 0.0
_last_loaded_ext  = 0.0

# ====== CSS: 히스토리(로그) 창을 드래그/스크롤 가능 박스로 고정 ======
CUSTOM_CSS = """
#history-box {
  max-height: 360px;
  min-height: 160px;
  overflow: auto;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 8px 10px;
  resize: vertical; /* 드래그로 높이 조절 */
  background: #fafafa;
}
"""

# ====== 유틸 ======
def _first_number(s):
    """'95.6%', '25 Lines', '1,024' → 95.6 / 25 / 1024"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).replace(",", "").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+", t)
    return float(m.group(0)) if m else None

def _num_series(series: pd.Series) -> pd.Series:
    return series.astype(str) \
        .str.replace(",", "", regex=False) \
        .str.replace("%", "", regex=False) \
        .str.extract(r"([-+]?\d*\.?\d+)")[0] \
        .astype(float)

def _norm_text(s: str) -> str:
    """라인 문자열 정규화: '25 Lines'→'25', 'Land & Win'→'land&win' 등"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).lower()
    t = t.replace(" ", "")
    t = t.replace("and", "&").replace("+", "&")
    t = t.replace("lines", "").replace("line", "").replace("라인", "")
    t = t.replace("ways", "").replace("way", "").replace("웨이", "")
    t = t.replace("landwin", "land&win")
    return t

def find_col(df: pd.DataFrame, cands):
    cols = {c.lower(): c for c in df.columns}
    for x in cands:
        if x and x.lower() in cols:
            return cols[x.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(x.lower() in lc for x in cands if x):
            return c
    return None

def find_multi_cols(df: pd.DataFrame, base_keywords, max_n=4):
    """theme1~4, feature1~4 같은 계열 칼럼 추출"""
    names = []
    for n in range(1, max_n+1):
        cand = find_col(df, [f"{k}{n}" for k in base_keywords])
        if cand and cand not in names:
            names.append(cand)
    for k in base_keywords:
        cand = find_col(df, [k])
        if cand and cand not in names:
            names.append(cand)
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in base_keywords) and c not in names:
            names.append(c)
        if len(names) >= max_n:
            break
    return names[:max_n]

def detect_schema_main(df: pd.DataFrame):
    # 내부 슬롯 스키마 (기존)
    return {
        "id":        find_col(df, ["game_id","slot_id","id","gameid","게임id","게임_id"]),
        "name":      find_col(df, ["title","name","게임명","slot","game"]),
        "type":      find_col(df, ["type","slot_type","게임타입","타입"]),  # lines/ways
        "overall_rtp": find_col(df, ["rtp","overall_rtp","total_rtp","전체 rtp"]),
        "base_rtp":    find_col(df, ["base_rtp","베이스 rtp"]),
        "free_rtp":    find_col(df, ["free_rtp","프리 rtp","freespin rtp"]),
        "volatility":  find_col(df, ["volatility","variance","편차","볼라"]),
        "hit_rate":    find_col(df, ["hit_rate","hit rate","히트","히트율","승률","확률"]),
        "lines":       find_col(df, ["line","lines","라인"]),
        "ways":        find_col(df, ["way","ways","웨이"]),
        "size":        find_col(df, ["size","grid","릴행","릴x행","릴수x행수"]),
        "reels":       find_col(df, ["reels","reel","릴","columns"]),
        "rows":        find_col(df, ["rows","row","행"]),
        "themes":      find_multi_cols(df, ["theme","테마"], max_n=4),
        "features":    find_multi_cols(df, ["feature","features","피쳐","특징"], max_n=4),
        "maker":       find_col(df, ["maker","provider","제작사","퍼블리셔"]),
    }

def detect_schema_ext(df: pd.DataFrame):
    # 외부 슬롯 스키마 (external_game_info.csv)
    return {
        "id":        find_col(df, ["game_id","slot_id","id","gameid","게임id","게임_id"]),
        "name":      find_col(df, ["title","name","게임명","slot","game"]),
        "ext_type":  find_col(df, ["rgs","sng","type","외부타입","외부 타입"]),  # rgs/sng
        "maker":     find_col(df, ["maker","provider","회사","제작사","퍼블리셔"]),
        "size":      find_col(df, ["size","grid","릴행","릴x행","릴수x행수"]),
        "lines":     find_col(df, ["line","lines","라인"]),
        "themes":    find_multi_cols(df, ["theme","테마"], max_n=4),
        "features":  find_multi_cols(df, ["feature","features","피쳐","특징"], max_n=4),
        # 참고: 외부 파일은 rtp/hit/volatility가 없을 수 있음
    }

# ====== 데이터 로드/준비 ======
def _read_csv(url: str, local_path: str|None = None):
    if local_path and os.path.exists(local_path):
        return pd.read_csv(local_path)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    try:
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.read_csv(io.StringIO(r.text), encoding="utf-8-sig")

def load_df(source: str = "main", force: bool = False) -> pd.DataFrame:
    global _df_cache_main, _df_cache_ext, _last_loaded_main, _last_loaded_ext
    now = time.time()
    if source == "main":
        if not force and _df_cache_main is not None and (now - _last_loaded_main) < TTL_SEC:
            return _df_cache_main
        df = _read_csv(CSV_URL_MAIN, CSV_LOCAL_MAIN)
        if df.empty or len(df.columns) == 0:
            raise RuntimeError("메인 CSV를 읽었지만 컬럼이 없습니다.")
        _df_cache_main, _last_loaded_main = df, now
        return df
    else:
        if not force and _df_cache_ext is not None and (now - _last_loaded_ext) < TTL_SEC:
            return _df_cache_ext
        df = _read_csv(CSV_URL_EXTERNAL, CSV_LOCAL_EXTERNAL)
        if df.empty or len(df.columns) == 0:
            raise RuntimeError("외부 CSV를 읽었지만 컬럼이 없습니다.")
        _df_cache_ext, _last_loaded_ext = df, now
        return df

def prepare_df_main(df_raw: pd.DataFrame):
    df = df_raw.copy()
    S = detect_schema_main(df)

    # 숫자 파생
    rtp_col = S["overall_rtp"] or S["base_rtp"] or S["free_rtp"]
    df["_rtp"]   = _num_series(df[rtp_col]) if rtp_col in df.columns else pd.NA
    df["_vola"]  = _num_series(df[S["volatility"]]) if S["volatility"] in df.columns else pd.NA
    df["_hit"]   = _num_series(df[S["hit_rate"]])   if S["hit_rate"] in df.columns   else pd.NA
    df["_lines"] = df[S["lines"]].apply(_first_number) if S["lines"] in df.columns else pd.NA
    df["_ways"]  = df[S["ways"]].apply(_first_number)  if S["ways"]  in df.columns else pd.NA

    # 퍼센트 단위 자동 보정
    for col in ["_rtp", "_hit"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.dropna().size > 0 and (s.dropna() <= 1.5).mean() > 0.8:
                df[col] = s * 100.0

    # 라인 텍스트 정규화(랜드앤윈 인식용)
    if S["lines"] in df.columns:
        df["_lines_txt"] = df[S["lines"]].apply(_norm_text)
    else:
        df["_lines_txt"] = ""

    # 사이즈 그리드 표준화
    if S["size"] in df.columns:
        df["_grid"] = (
            df[S["size"]].astype(str)
            .str.extract(r"(\d+\s*[xX×]\s*\d+)")[0]
            .str.replace(" ", "", regex=False).str.lower()
        )
    elif S["reels"] in df.columns and S["rows"] in df.columns:
        df["_grid"] = (
            df[S["reels"]].astype(str).str.extract(r"(\d+)")[0]
            + "x" +
            df[S["rows"]].astype(str).str.extract(r"(\d+)")[0]
        ).str.lower()
    else:
        df["_grid"] = pd.NA

    return df, S

def prepare_df_ext(df_raw: pd.DataFrame):
    df = df_raw.copy()
    S = detect_schema_ext(df)

    # 숫자 파생 (ext에는 rtp/hit/vola가 없을 수 있음)
    if S["lines"] in df.columns:
        df["_lines"] = df[S["lines"]].apply(_first_number)
        df["_lines_txt"] = df[S["lines"]].apply(_norm_text)
    else:
        df["_lines"] = pd.NA
        df["_lines_txt"] = ""

    if S["size"] in df.columns:
        df["_grid"] = (
            df[S["size"]].astype(str)
            .str.extract(r"(\d+\s*[xX×]\s*\d+)")[0]
            .str.replace(" ", "", regex=False).str.lower()
        )
    else:
        df["_grid"] = pd.NA

    # ext_type 정규화(rgs/sng)
    if S["ext_type"] in df.columns:
        df["_ext_type"] = df[S["ext_type"]].astype(str).str.lower().str.extract(r"(rgs|sng)")[0]
    else:
        df["_ext_type"] = pd.NA

    return df, S

# ====== 파서 ======
NUM   = r"(?P<num>\d+(?:\.\d+)?)"
RANGE = r"(?P<n1>\d+(?:\.\d+)?)[\s]*(?:~|-|to|–|—|까지)\s*(?P<n2>\d+(?:\.\d+)?)"
SIZEP = r"(?P<r>\d+)\s*[xX×]\s*(?P<c>\d+)"

FIELD_ALIASES = {
    "rtp":        [r"\brtp\b", r"\boverall\s*rtp\b", r"\btotal\s*rtp\b", r"전체\s*rtp"],
    "volatility": [r"편차", r"\bvolatility\b", r"\bvariance\b", r"볼라"],
    "hit_rate":   [r"히트(율)?", r"hit\s*rate", r"승률", r"확률"],
    "lines":      [r"라인", r"\blines?\b"],
    "ways":       [r"웨이",  r"\bways?\b"],
}

CMP_WORD = {"이상": ">=", "이하": "<=", "초과": ">", "미만": "<", "같음": "=="}
LANDWIN_WORDS = ["land&win","land and win","랜드앤윈","랜앤윈","hold&win","hold and win"]

# 외부 슬롯 인식용
EXTERNAL_SYNONYMS = [
    "외부", "외부슬롯", "외부 슬롯", "외부 타입", "외부타입",
    "타사", "타 회사", "타회사", "외 슬롯", "외 타입"
]

STOPWORDS = set("""
이상 이하 초과 미만 부터 까지 사이 그리고 이며 이고 은 는 이 가 을 를 에 의 도 만 와 과 과의 및 에서 으로 로 도는 는데 한데 
and & between to the a of slot slots 슬롯 찾아줘 찾아 줘 찾아 알려줘 알려 줘 타입 type size 사이즈
rtp volatility variance 편차 볼라 히트 히트율 승률 확률 line lines 라인 way ways 웨이 land&win 랜드앤윈
rgs sng 외부 외부슬롯 외부타입 외부 타입 타사 타 회사 타회사 외 슬롯 외 타입
""".split())

def _normalize_query_text(q: str) -> str:
    q = q.strip()
    q = q.replace("코인 그랩", "코인그랩").replace("coin grab", "coingrab")
    return q

def _extract_numeric(field_key, text):
    res = []
    aliases = FIELD_ALIASES.get(field_key, [])
    # 1) 범위
    rp = rf"({'|'.join(aliases)})\s*(?:{RANGE})"
    for m in re.finditer(rp, text, flags=re.I):
        a, b = float(m.group("n1")), float(m.group("n2"))
        res.append(("between", (min(a,b), max(a,b))))
    if res: return res
    # 2-a) 비교/기호 (연산자→숫자)
    cw = "|".join(CMP_WORD.keys()); ops = r"[<>]=?|=="
    cp1 = rf"({'|'.join(aliases)})\s*(?:({ops}|{cw}))\s*{NUM}%?"
    for m in re.finditer(cp1, text, flags=re.I):
        op = CMP_WORD.get(m.group(2), m.group(2))
        val = float(m.group("num"))
        res.append((op, val))
    if res: return res
    # 2-b) 비교/기호 (숫자→연산자)
    cp2 = rf"({'|'.join(aliases)})\s*{NUM}%?\s*(?:({ops}|{cw}))"
    for m in re.finditer(cp2, text, flags=re.I):
        op = CMP_WORD.get(m.group(2), m.group(2))
        val = float(m.group("num"))
        res.append((op, val))
    if res: return res
    # 3) 단일값 → [v, v+0.5]
    sp = rf"({'|'.join(aliases)})\s*{NUM}%?"
    for m in re.finditer(sp, text, flags=re.I):
        v = float(m.group("num"))
        res.append(("between", (v, v+0.5)))
    return res

def parse_query(q: str):
    q = _normalize_query_text(q)
    low = q.lower()
    out = {
        "size": None, "rtp": [], "volatility": [], "hit_rate": [],
        "lines": [], "ways": [], "keywords": [], "landwin": False,
        "external_only": False, "ext_type": None  # rgs/sng
    }
    if not q: return out

    # 외부 슬롯 라우팅
    if any(s in low for s in [s.lower() for s in EXTERNAL_SYNONYMS]):
        out["external_only"] = True
    # rgs/sng 타입 추출
    if re.search(r"\brgs\b", low): out["ext_type"] = "rgs"
    if re.search(r"\bsng\b", low): out["ext_type"] = "sng"

    # 사이즈 5x3
    m = re.search(SIZEP, q, flags=re.I)
    if m:
        out["size"] = (int(m.group("r")), int(m.group("c")))

    # 수치 조건(내부 슬롯에만 사용됨)
    out["rtp"]        = _extract_numeric("rtp", q)
    out["volatility"] = _extract_numeric("volatility", q)
    out["hit_rate"]   = _extract_numeric("hit_rate", q)

    # 25 라인 / 243 ways
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(lines?|라인)\b", q, flags=re.I)
    if m: out["lines"] = [("==", float(m.group(1)))]
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(ways?|웨이)\b", q, flags=re.I)
    if m: out["ways"]  = [("==", float(m.group(1)))]

    # 랜드앤윈
    if any(w in low for w in [w.lower() for w in LANDWIN_WORDS]):
        out["landwin"] = True

    # 키워드 (필드명/불용어/구두점 제거)
    cleaned = re.sub(SIZEP, " ", q, flags=re.I)
    cleaned = re.sub(r"\d+(\.\d+)?\s*(%|라인|웨이)?", " ", cleaned)
    cleaned = re.sub(r"[><=~\-:]+", " ", cleaned)
    cleaned = cleaned.replace(",", " ").replace("/", " ").replace("|", " ").replace("(", " ").replace(")", " ")
    toks = [t.strip().lower() for t in re.split(r"\s+", cleaned) if t.strip()]
    toks = [re.sub(r"^\W+|\W+$", "", t) for t in toks]  # 양끝 구두점 제거
    toks = [t for t in toks if t and t not in STOPWORDS]
    out["keywords"] = toks
    return out

# ====== 필터 ======
def apply_numeric(df: pd.DataFrame, series_name: str, clauses):
    if not clauses or series_name not in df.columns:
        return pd.Series(True, index=df.index)
    s = pd.to_numeric(df[series_name], errors="coerce")
    m = pd.Series(True, index=df.index)
    for op, val in clauses:
        if op == "between":
            a, b = val
            m &= (s >= a) & (s <= b)
        elif op == ">=": m &= (s >= val)
        elif op == "<=": m &= (s <= val)
        elif op == ">":  m &= (s >  val)
        elif op == "<":  m &= (s <  val)
        elif op == "==": m &= (s == val)
    return m

def filter_size(df: pd.DataFrame, size_tuple):
    if not size_tuple: return pd.Series(True, index=df.index)
    r, c = size_tuple
    return df["_grid"].astype(str).str.fullmatch(fr"{r}[xX×]{c}", case=False, na=False)

def keyword_mask_text(df: pd.DataFrame, text_cols, toks):
    if not toks:
        return pd.Series(True, index=df.index)
    m = pd.Series(True, index=df.index)
    # 키워드 AND
    for t in toks:
        tmask = pd.Series(False, index=df.index)
        for c in text_cols:
            tmask |= df[c].astype(str).str.contains(re.escape(t), case=False, na=False)
        m &= tmask
    return m

# ====== 결과 컬럼 선택 ======
def build_display_main(df: pd.DataFrame, S, message: str, q=None):
    idcol   = S["id"] or S["name"]
    namecol = S["name"]
    typecol = S["type"]

    rtp_col = S["overall_rtp"] or S["base_rtp"] or S["free_rtp"]
    vola_col= S["volatility"]
    hit_col = S["hit_rate"]
    line_col= S["lines"]
    ways_col= S["ways"]
    size_col= S["size"]

    themes = [c for c in (S.get("themes") or []) if c in df.columns]
    feats  = [c for c in (S.get("features") or []) if c in df.columns]

    msg_l = (message or "").lower()

    wants_rtp   = ("rtp" in msg_l) or (q and q.get("rtp"))
    wants_hit   = (re.search(r"(히트|hit)", message or "", flags=re.I) is not None) or (q and q.get("hit_rate"))
    wants_vol   = (re.search(r"(편차|volatility|variance|볼라)", message or "", flags=re.I) is not None) or (q and q.get("volatility"))
    wants_lines = (re.search(r"(라인|lines?|land&win|랜드앤윈)", message or "", flags=re.I) is not None) or (q and q.get("lines"))
    wants_size  = ("사이즈" in (message or "")) or (re.search(SIZEP, message or "") is not None) or (q and q.get("size"))
    wants_theme = ("테마" in (message or "")) or ("동화" in (message or ""))
    wants_feat  = ("피쳐" in (message or "")) or ("특징" in (message or "")) or ("feature" in msg_l)

    # 키워드가 theme/feature 값과 매칭되면 자동 노출
    if not df.empty and q and q.get("keywords"):
        kw = q["keywords"]
        if themes:
            vals = df[themes].fillna("").astype(str).agg(" ".join, axis=1)
            if any(vals.str.contains(re.escape(t), case=False, na=False).any() for t in kw):
                wants_theme = True
        if feats:
            valsf = df[feats].fillna("").astype(str).agg(" ".join, axis=1)
            if any(valsf.str.contains(re.escape(t), case=False, na=False).any() for t in kw):
                wants_feat = True

    cols = []
    if idcol: cols.append(idcol)
    elif namecol: cols.append(namecol)
    if typecol: cols.append(typecol)  # lines/ways

    if wants_rtp and rtp_col: cols.append(rtp_col)
    if wants_hit and hit_col: cols.append(hit_col)
    if wants_vol and vola_col: cols.append(vola_col)
    if wants_lines:
        if line_col: cols.append(line_col)
        elif ways_col: cols.append(ways_col)
    if wants_size:
        cols.append(size_col if size_col else "_grid")
    if wants_theme: cols += themes
    if wants_feat:  cols += feats

    # 기본 요약 컬럼 보충
    if len(cols) <= 2:
        for c in [rtp_col, vola_col, hit_col, (line_col or ways_col), (size_col or "_grid")]:
            if c and c not in cols and (c in df.columns or c == "_grid"):
                cols.append(c)

    # 실제 존재만 + 중복 제거
    seen, final_cols = set(), []
    for c in cols:
        if c and (c == "_grid" or c in df.columns) and c not in seen:
            seen.add(c); final_cols.append(c)

    rename = {
        (idcol or ""): "game_id" if idcol else "게임명",
        (namecol or ""): "게임명",
        (typecol or ""): "type",           # lines/ways
        (rtp_col or ""): "rtp",
        (vola_col or ""): "volatility",
        (hit_col or ""): "hit_rate",
        (line_col or ""): "line",
        (ways_col or ""): "ways",
        (size_col or ""): "size",
        "_grid": "size",
    }
    out = df.loc[:, [c for c in final_cols if c != "_grid"]].copy()
    if "_grid" in final_cols and "_grid" in df.columns and "size" not in out.columns:
        out.insert(len(out.columns), "size", df["_grid"])
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns})
    return out

def build_display_ext(df: pd.DataFrame, S, message: str, q=None):
    idcol   = S["id"] or S["name"]
    namecol = S["name"]
    extcol  = S["ext_type"]  # rgs/sng
    maker   = S["maker"]
    line_col= S["lines"]
    size_col= S["size"]

    themes = [c for c in (S.get("themes") or []) if c in df.columns]
    feats  = [c for c in (S.get("features") or []) if c in df.columns]

    msg_l = (message or "").lower()
    wants_lines = (re.search(r"(라인|lines?)", message or "", flags=re.I) is not None) or (q and q.get("lines"))
    wants_size  = ("사이즈" in (message or "")) or (re.search(SIZEP, message or "") is not None) or (q and q.get("size"))
    wants_theme = ("테마" in (message or "")) or ("동화" in (message or ""))
    wants_feat  = ("피쳐" in (message or "")) or ("특징" in (message or "")) or ("feature" in msg_l)

    cols = []
    if idcol: cols.append(idcol)
    if extcol: cols.append(extcol)     # rgs/sng
    if maker:  cols.append(maker)      # 제작사/회사
    if wants_lines and line_col: cols.append(line_col)
    if wants_size:
        cols.append(size_col if size_col else "_grid")
    if wants_theme: cols += themes
    if wants_feat:  cols += feats

    # 기본 보충
    if len(cols) <= 2:
        for c in [line_col, (size_col or "_grid")]:
            if c and c not in cols and (c in df.columns or c == "_grid"):
                cols.append(c)

    # dedupe
    seen, final_cols = set(), []
    for c in cols:
        if c and (c == "_grid" or c in df.columns) and c not in seen:
            seen.add(c); final_cols.append(c)

    rename = {
        (idcol or ""): "game_id",
        (namecol or ""): "게임명",
        (extcol or ""): "ext_type",
        (maker or ""): "maker",
        (line_col or ""): "line",
        (size_col or ""): "size",
        "_grid": "size",
    }
    out = df.loc[:, [c for c in final_cols if c != "_grid"]].copy()
    # 이름 보조 컬럼이 있다면 뒤에 추가
    if namecol and namecol in df.columns and namecol not in out.columns:
        out.insert(1, "게임명", df[namecol])
    if "_grid" in final_cols and "_grid" in df.columns and "size" not in out.columns:
        out.insert(len(out.columns), "size", df["_grid"])
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns})
    return out

# ====== 검색 실행 ======
def run_search(message: str):
    q = parse_query(message)

    # 외부 슬롯 플로우
    if q["external_only"] or q["ext_type"]:
        df_raw = load_df("external")
        df, S = prepare_df_ext(df_raw)

        # rgs/sng 필수: 없으면 공지 후 빈 결과
        if not q["ext_type"]:
            view = pd.DataFrame(columns=["game_id","ext_type","maker","size","line"])
            info = "외부 슬롯은 rgs 또는 sng 타입을 반드시 포함해야 검색됩니다. 예) 'rgs, 5x4', 'sng, 5x3', 'sng 패스트크레딧'."
            return info, view

        m = pd.Series(True, index=df.index)
        # ext_type
        m &= (df["_ext_type"] == q["ext_type"])
        # size
        m &= filter_size(df, q["size"])
        # 라인 숫자 필터 (라인/ways 개념 없음 → line 숫자만)
        if "_lines" in df.columns:
            m &= apply_numeric(df, "_lines", q["lines"])
        # 키워드: name/id/maker/themes/features 에서 AND 검색
        text_cols = []
        for key in ["name","id","maker"]:
            col = S.get(key)
            if col in df.columns: text_cols.append(col)
        for c in (S.get("themes") or []):
            if c in df.columns: text_cols.append(c)
        for c in (S.get("features") or []):
            if c in df.columns: text_cols.append(c)
        m &= keyword_mask_text(df, text_cols, q["keywords"])

        res = df.loc[m].copy()
        # 정렬: maker, name
        sort_cols = [c for c in [S.get("maker"), S.get("name")] if c in res.columns]
        if sort_cols:
            res = res.sort_values(sort_cols, ascending=[True]*len(sort_cols), na_position="last")

        view = build_display_ext(res, S, message, q=q).head(200).reset_index(drop=True)
        return f"총 {len(view)}개 결과 (외부 슬롯: {q['ext_type']})", view

    # 내부 슬롯 플로우(기존)
    df_raw = load_df("main")
    df, S = prepare_df_main(df_raw)

    m = pd.Series(True, index=df.index)
    m &= filter_size(df, q["size"])
    if "_rtp" in df.columns:   m &= apply_numeric(df, "_rtp",  q["rtp"])
    if "_vola" in df.columns:  m &= apply_numeric(df, "_vola", q["volatility"])
    if "_hit" in df.columns:   m &= apply_numeric(df, "_hit",  q["hit_rate"])
    if "_lines" in df.columns: m &= apply_numeric(df, "_lines", q["lines"])
    if "_ways"  in df.columns: m &= apply_numeric(df, "_ways",  q["ways"])

    # 텍스트 키워드
    text_cols = []
    for k in ["name", "id", "type", "maker"]:
        if S.get(k) in df.columns:
            text_cols.append(S[k])
    for c in (S.get("themes") or []):
        if c in df.columns: text_cols.append(c)
    for c in (S.get("features") or []):
        if c in df.columns: text_cols.append(c)
    m &= keyword_mask_text(df, text_cols, q["keywords"])
    # 랜드앤윈
    m &= df["_lines_txt"].astype(str).str.contains("land&win", na=False) if q["landwin"] else m

    res = df.loc[m].copy()

    # 정렬: rtp/편차/히트율/이름
    sort_cols = [c for c in ["_rtp","_vola","_hit", S["name"]] if c in res.columns]
    if sort_cols:
        res = res.sort_values(sort_cols, ascending=[False, False, False, True][:len(sort_cols)], na_position="last")

    view = build_display_main(res, S, message, q=q).head(200).reset_index(drop=True)
    info = f"총 {len(view)}개 결과"
    return info, view

# ====== UI ======
def render_history(hist):
    lines = []
    for u, a in hist:
        lines.append(f"**You:** {u}")
        lines.append(f"**Bot:** {a}")
        lines.append("---")
    return "\n".join(lines) if hist else "_대화를 시작해보세요._"

with gr.Blocks(css=CUSTOM_CSS, title="Kanana 슬롯 검색 — 슬롯서치완성1.3-external") as demo:
    gr.Markdown("### Kanana 슬롯 검색 — 슬롯서치완성1.3-external (GitHub CSV / data 브랜치)")

    with gr.Row():
        # 좌측: 히스토리(드래그/스크롤 고정)
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("**🗒️ 로그 / 검색 기록**")
            history = gr.State([])
            history_md = gr.Markdown(render_history([]), elem_id="history-box")
            clear_btn = gr.Button("로그 지우기")

        # 우측: 검색/결과
        with gr.Column(scale=3):
            query = gr.Textbox(
                placeholder="예) rgs 5x4 / sng 5x3 / sng 타입 5x3 / 외부 슬롯 패스트크레딧 / 25라인 / 243 ways / 5x3 / 편차 11 / rtp 95%",
                label="질문/조건"
            )
            with gr.Row():
                search_btn  = gr.Button("검색")
                preview_btn = gr.Button("CSV 미리보기(내부)")
                preview_ext = gr.Button("CSV 미리보기(외부)")
                refresh_btn = gr.Button("데이터 새로고침")

            status_md = gr.Markdown()
            results_df = gr.Dataframe(label="검색 결과", interactive=False, height=520, wrap=True)

    # 핸들러
    def on_search(msg, hist):
        try:
            info, df = run_search(msg)
        except Exception as e:
            info, df = f"❌ 에러: {e}", pd.DataFrame()
        new_hist = (hist or []) + [(msg, info)]
        return "", new_hist, render_history(new_hist), df

    def on_preview():
        try:
            df_raw = load_df("main")
            df, S = prepare_df_main(df_raw)
            preview_cols = build_display_main(df, S, "미리보기").head(20).reset_index(drop=True)
            return preview_cols
        except Exception as e:
            return pd.DataFrame({"오류":[str(e)]})

    def on_preview_ext():
        try:
            df_raw = load_df("external")
            df, S = prepare_df_ext(df_raw)
            preview_cols = build_display_ext(df, S, "미리보기").head(20).reset_index(drop=True)
            return preview_cols
        except Exception as e:
            return pd.DataFrame({"오류":[str(e)]})

    def on_refresh():
        try:
            load_df("main", force=True)
            load_df("external", force=True)
            return "✅ 최신 CSV로 새로 불러왔습니다. (내부/외부)"
        except Exception as e:
            return f"❌ 새로고침 실패: {e}"

    def on_clear():
        return [], render_history([])

    # 바인딩
    query.submit(on_search, [query, history], [query, history, history_md, results_df])
    search_btn.click(on_search, [query, history], [query, history, history_md, results_df])
    preview_btn.click(on_preview, None, results_df)
    preview_ext.click(on_preview_ext, None, results_df)
    refresh_btn.click(lambda: on_refresh(), None, status_md)
    clear_btn.click(on_clear, None, [history, history_md])

# 엔드포인트 활성화
demo.queue()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.getenv("PORT", 7860)),
                share=True,
                show_error=True)
