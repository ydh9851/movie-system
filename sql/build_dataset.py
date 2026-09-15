# -*- coding: utf-8 -*-
"""
数据集替换 / 重建脚本
=====================
把 Kaggle 原始数据集转换成本工程各算法模块要求的文件格式，写入 data/ 目录。

数据来源（data/raw/）：
    the-movies-dataset/      Kaggle: rounakbanik/the-movies-dataset
        movies_metadata.csv  45k 电影元数据
        credits.csv          演职员（cast / crew）
        keywords.csv         关键词
    ml-25m/                  MovieLens 25M（官方 ml-25m）
        movies.csv / ratings.csv / links.csv

产出（data/）：
    train.csv                            票房预测训练集（含 revenue）
    test.csv                             票房预测待预测集（无 revenue）
    sample_submission.csv                提交样例
    recommendation/tmdb_5000_movies.csv  推荐用 TMDB 元数据（按 vote_count 取 Top N）
    recommendation/tmdb_5000_credits.csv 推荐用演职员（恰好 4 列：movie_id,title,cast,crew）
    recommendation/personal/*.csv        MovieLens 采样后的个性化推荐数据 + 划分好的 train/test
    prediction/TrainAdditionalFeatures.csv   附加特征（imdb_id,popularity2,rating,totalVotes）
    prediction/TestAdditionalFeatures.csv
    prediction/release_dates_per_country.csv 上映国家表（见下方说明）

重要格式约定（换数据集时最容易踩的坑）：
    1. train.csv / test.csv 里的列表字段（genres / cast / crew / Keywords ...）必须是
       Python repr 字符串（单引号、True/False 大写），因为 preprocess.py 用 ast.literal_eval 解析；
       原始 Kaggle 数据里可能混有 JSON 的 false / true / null，直接用会抛异常。
       本脚本统一用 repr() 重新序列化，保证两种解析器都能读。
    2. recommendation/tmdb_5000_credits.csv 必须**恰好 4 列**，顺序固定为
       movie_id, title, cast, crew —— naive_recommender 是按位置重命名列的。
    3. eda.py 里 release_dates_per_country.csv 是**按行号位置**与 train.csv 对齐的
       （release_dates["id"] = range(1, len+1)），所以本表行序必须与 train.csv 完全一致。

⚠ 关于 release_dates_per_country.csv：
    原始版本记录的是「各片在各国是否有院线上映 + 上映国家数」，这份数据来自 TMDB 的
    release_dates 接口，the-movies-dataset 里**没有**对应字段，无法忠实重建。
    本脚本改为用 movies_metadata 的 production_countries 重建：
        国家列 = 该片是否由该国制作（1/0）
        theatrical = 制作国家数量
    因此 eda.py 中 fig17 的语义随之由「上映规模」变为「制作国家数」。

用法：
    .venv\\Scripts\\python.exe sql/build_dataset.py
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(SYSTEM_ROOT, "data", "raw")
DATA_DIR = os.path.join(SYSTEM_ROOT, "data")
REC_DIR = os.path.join(DATA_DIR, "recommendation")
PERSONAL_DIR = os.path.join(REC_DIR, "personal")
PRED_DIR = os.path.join(DATA_DIR, "prediction")

TMDB_RAW = os.path.join(RAW_DIR, "the-movies-dataset")
ML_RAW = os.path.join(RAW_DIR, "ml-25m")

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
REC_TOP_N = 10000        # 推荐元数据保留的电影数（按 vote_count 取 Top N）
                         # 说明：naive 推荐会算 N×N 稠密相似度矩阵，
                         #       N=4806 约 185MB / 1.3s；N=10000 约 800MB / 6s；
                         #       N=45000 约 16GB，必然内存溢出。
ML_TOP_USERS = 800       # MovieLens 采样：保留评分最多的用户数
ML_TOP_MOVIES = 3000     # MovieLens 采样：保留被评分最多的电影数
ML_TRAIN_RATIO = 0.85    # 与 personal_recommender/test.py 的 threshold 保持一致
RANDOM_SEED = 42

# 与原始 release_dates_per_country.csv 相同的国家列
COUNTRY_COLS = ["IT", "DE", "FR", "UA", "LV", "US", "PT", "CZ", "SE", "DK", "KZ", "NL",
                "RU", "HK", "CL", "TW", "IE", "NO", "GR", "GB", "AU", "CA", "AR", "BE",
                "TR", "FI", "TH", "IS", "SK", "IL", "PL", "ID", "MX", "AT", "SI", "HU",
                "KW", "BG", "PE"]


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def parse_list(v):
    """把 Kaggle 里各式各样的列表字段（JSON / repr / 单字典）安全解析为 list。"""
    if v is None:
        return []
    if isinstance(v, float) and pd.isna(v):
        return []
    if isinstance(v, (list, dict)):
        return v if isinstance(v, list) else [v]
    if not isinstance(v, str) or not v.strip():
        return []
    s = v.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            x = parser(s)
        except Exception:
            continue
        if isinstance(x, dict):
            return [x]
        if isinstance(x, list):
            return x
        return []
    return []


def dump_list(lst):
    """统一序列化为 Python repr —— ast.literal_eval 与 json 兜底解析都能读。"""
    return repr(lst)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def text_col(s):
    """文本列：NaN -> 空串，并去掉首尾空白。"""
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def iso_codes(lst):
    """从 production_countries 的 [{iso_3166_1, name}] 中取出国家代码。"""
    out = []
    for d in lst:
        if isinstance(d, dict) and d.get("iso_3166_1"):
            out.append(str(d["iso_3166_1"]).upper())
    return out


# ---------------------------------------------------------------------------
# 1. 读取 Kaggle 原始数据
# ---------------------------------------------------------------------------
def load_raw():
    log("[1/8] 读取 Kaggle 原始数据 ...")

    meta = pd.read_csv(os.path.join(TMDB_RAW, "movies_metadata.csv"), low_memory=False)
    log(f"      movies_metadata.csv  {meta.shape}")

    credits = pd.read_csv(os.path.join(TMDB_RAW, "credits.csv"))
    log(f"      credits.csv          {credits.shape}")

    keywords = pd.read_csv(os.path.join(TMDB_RAW, "keywords.csv"))
    log(f"      keywords.csv         {keywords.shape}")

    # --- 统一 id 列并去重（3 行脏数据 id 是日期字符串，必须丢掉）---
    raw_rows = len(meta)
    meta["tmdb_id"] = to_num(meta["id"])
    bad = meta["tmdb_id"].isna().sum()
    meta = meta[meta["tmdb_id"].notna()].copy()
    meta["tmdb_id"] = meta["tmdb_id"].astype(int)
    before = len(meta)
    meta = meta.drop_duplicates(subset=["tmdb_id"], keep="first").copy()
    log(f"      丢弃 id 非数值行 {bad} 行，重复 id {before - len(meta)} 行，"
        f"有效电影 {len(meta)} 部（原始 {raw_rows}）")

    for df, name in ((credits, "credits"), (keywords, "keywords")):
        df["tmdb_id"] = to_num(df["id"])
        df.dropna(subset=["tmdb_id"], inplace=True)
        df["tmdb_id"] = df["tmdb_id"].astype(int)
        df.drop_duplicates(subset=["tmdb_id"], keep="first", inplace=True)
        log(f"      {name} 去重后 {len(df)} 条")

    # --- 解析列表字段 ---
    list_cols = ["belongs_to_collection", "genres", "production_companies",
                 "production_countries", "spoken_languages"]
    for c in list_cols:
        meta[c + "_p"] = meta[c].apply(parse_list)
    credits["cast_p"] = credits["cast"].apply(parse_list)
    credits["crew_p"] = credits["crew"].apply(parse_list)
    keywords["keywords_p"] = keywords["keywords"].apply(parse_list)

    # --- 数值列 ---
    for src, dst in (("budget", "budget_n"), ("revenue", "revenue_n"),
                     ("popularity", "popularity_n"), ("runtime", "runtime_n"),
                     ("vote_average", "vote_average_n"), ("vote_count", "vote_count_n")):
        meta[dst] = to_num(meta[src])

    # 合并 keywords / credits
    meta = meta.merge(keywords[["tmdb_id", "keywords_p"]], on="tmdb_id", how="left")
    meta = meta.merge(credits[["tmdb_id", "cast_p", "crew_p"]], on="tmdb_id", how="left")
    meta["keywords_p"] = meta["keywords_p"].apply(lambda x: x if isinstance(x, list) else [])
    meta["cast_p"] = meta["cast_p"].apply(lambda x: x if isinstance(x, list) else [])
    meta["crew_p"] = meta["crew_p"].apply(lambda x: x if isinstance(x, list) else [])

    return meta


# ---------------------------------------------------------------------------
# 2. 票房预测：train / test / sample_submission
# ---------------------------------------------------------------------------
TRAIN_COLS = ["id", "belongs_to_collection", "budget", "genres", "homepage", "imdb_id",
              "original_language", "original_title", "overview", "popularity", "poster_path",
              "production_companies", "production_countries", "release_date", "runtime",
              "spoken_languages", "status", "tagline", "title", "Keywords", "cast", "crew"]


def build_boxoffice(meta):
    log()
    log("[2/8] 构建票房预测数据集 train.csv / test.csv / sample_submission.csv ...")

    has_budget = meta["budget_n"] > 0
    has_revenue = meta["revenue_n"] > 0

    train = meta[has_budget & has_revenue].copy()
    test = meta[has_budget & ~has_revenue].copy()
    log(f"      候选：有预算 {int(has_budget.sum())} 部，其中有票房 {int((has_budget & has_revenue).sum())} 部")

    # 排序：按票房降序，保证结果稳定可复现
    train = train.sort_values("revenue_n", ascending=False).reset_index(drop=True)
    test = test.sort_values("tmdb_id").reset_index(drop=True)

    n_train = len(train)
    train["id"] = np.arange(1, n_train + 1)
    test["id"] = np.arange(n_train + 1, n_train + len(test) + 1)
    log(f"      train.csv {len(train)} 行（原 3000）  test.csv {len(test)} 行（原 4404）")

    def assemble(df, with_revenue):
        out = pd.DataFrame(index=df.index)
        out["id"] = df["id"].astype(int)
        out["belongs_to_collection"] = df["belongs_to_collection_p"].apply(dump_list)
        out["budget"] = df["budget_n"].round().astype("Int64")
        out["genres"] = df["genres_p"].apply(dump_list)
        out["homepage"] = text_col(df["homepage"])
        out["imdb_id"] = text_col(df["imdb_id"])
        out["original_language"] = text_col(df["original_language"])
        out["original_title"] = text_col(df["original_title"])
        out["overview"] = text_col(df["overview"])
        out["popularity"] = df["popularity_n"].round(6)
        out["poster_path"] = text_col(df["poster_path"])
        out["production_companies"] = df["production_companies_p"].apply(dump_list)
        out["production_countries"] = df["production_countries_p"].apply(dump_list)
        out["release_date"] = text_col(df["release_date"])
        out["runtime"] = df["runtime_n"].round().astype("Int64")
        out["spoken_languages"] = df["spoken_languages_p"].apply(dump_list)
        out["status"] = text_col(df["status"])
        out["tagline"] = text_col(df["tagline"])
        out["title"] = text_col(df["title"])
        out["Keywords"] = df["keywords_p"].apply(dump_list)
        out["cast"] = df["cast_p"].apply(dump_list)
        out["crew"] = df["crew_p"].apply(dump_list)
        if with_revenue:
            out["revenue"] = df["revenue_n"].round().astype("Int64")
        return out

    train_out = assemble(train, True)
    test_out = assemble(test, False)
    sub_out = pd.DataFrame({"id": test_out["id"].astype(int), "revenue": 1000000})
    return train, test, train_out, test_out, sub_out


# ---------------------------------------------------------------------------
# 3. 推荐：tmdb_5000_movies.csv / tmdb_5000_credits.csv
# ---------------------------------------------------------------------------
REC_MOVIE_COLS = ["budget", "genres", "homepage", "id", "keywords", "original_language",
                  "original_title", "overview", "popularity", "production_companies",
                  "production_countries", "release_date", "revenue", "runtime",
                  "spoken_languages", "status", "tagline", "title", "vote_average", "vote_count"]


def build_recommendation_meta(meta):
    log()
    log(f"[3/8] 构建推荐用 TMDB 元数据（按 vote_count 取 Top {REC_TOP_N}）...")

    pool = meta[(meta["vote_count_n"] > 0) & (text_col(meta["overview"]) != "")].copy()
    pool = pool.sort_values("vote_count_n", ascending=False).head(REC_TOP_N)
    pool = pool.reset_index(drop=True)  # 先定序，movies / credits 共用，保证两者逐行对齐
    log(f"      候选 {len(pool)} 部（要求 vote_count>0 且 overview 非空）")

    movies = pd.DataFrame(index=pool.index)
    movies["budget"] = pool["budget_n"].round().astype("Int64")
    movies["genres"] = pool["genres_p"].apply(dump_list)
    movies["homepage"] = text_col(pool["homepage"])
    movies["id"] = pool["tmdb_id"].astype(int)
    movies["keywords"] = pool["keywords_p"].apply(dump_list)
    movies["original_language"] = text_col(pool["original_language"])
    movies["original_title"] = text_col(pool["original_title"])
    movies["overview"] = text_col(pool["overview"])
    movies["popularity"] = pool["popularity_n"].round(6)
    movies["production_companies"] = pool["production_companies_p"].apply(dump_list)
    movies["production_countries"] = pool["production_countries_p"].apply(dump_list)
    movies["release_date"] = text_col(pool["release_date"])
    movies["revenue"] = pool["revenue_n"].round().astype("Int64")
    movies["runtime"] = pool["runtime_n"].round().astype("Int64")
    movies["spoken_languages"] = pool["spoken_languages_p"].apply(dump_list)
    movies["status"] = text_col(pool["status"])
    movies["tagline"] = text_col(pool["tagline"])
    movies["title"] = text_col(pool["title"])
    movies["vote_average"] = pool["vote_average_n"].round(6)
    movies["vote_count"] = pool["vote_count_n"].round().astype("Int64")
    movies = movies[REC_MOVIE_COLS].reset_index(drop=True)

    # credits.csv 必须恰好 4 列，顺序 movie_id,title,cast,crew（naive_recommender 按位置重命名）
    credits = pd.DataFrame({
        "movie_id": pool["tmdb_id"].astype(int),
        "title": text_col(pool["title"]),
        "cast": pool["cast_p"].apply(dump_list),
        "crew": pool["crew_p"].apply(dump_list),
    })
    log(f"      tmdb_5000_movies.csv {movies.shape}   tmdb_5000_credits.csv {credits.shape}")
    return movies, credits


# ---------------------------------------------------------------------------
# 4. 个性化推荐：MovieLens 采样
# ---------------------------------------------------------------------------
def build_personal():
    log()
    log(f"[4/8] 构建个性化推荐数据（MovieLens 采样：Top {ML_TOP_USERS} 用户 × Top {ML_TOP_MOVIES} 电影）...")

    ratings = pd.read_csv(os.path.join(ML_RAW, "ratings.csv"))
    movies = pd.read_csv(os.path.join(ML_RAW, "movies.csv"))
    links = pd.read_csv(os.path.join(ML_RAW, "links.csv"))
    log(f"      ml-25m 原始：ratings {len(ratings)} / movies {len(movies)} / links {len(links)}")

    top_users = ratings["userId"].value_counts().head(ML_TOP_USERS).index
    sub = ratings[ratings["userId"].isin(top_users)]
    top_movies = sub["movieId"].value_counts().head(ML_TOP_MOVIES).index
    sub = sub[sub["movieId"].isin(top_movies)].copy()

    # 把采样后的 userId 重新编号成连续的 1..N（按评分条数降序）。
    # 原因：ml-25m 的 userId 是 1~162541 的原始编号，采样后既不连续也没有 1 号用户；
    #       而前端 recommend/index.vue 的「用户ID」默认值 11 约定 = MovieLens 用户 1
    #       （后端 t_user.id = MovieLens userId + 10），不重编号会出现
    #       "User N is not part of the trainset" 报错。
    order = sub["userId"].value_counts().index.tolist()
    remap = {old: i + 1 for i, old in enumerate(order)}
    sub["userId"] = sub["userId"].map(remap).astype("int64")
    sub = sub.sort_values(["userId", "movieId"]).reset_index(drop=True)
    log(f"      采样后：{len(sub)} 条评分 / {sub['userId'].nunique()} 用户 / {sub['movieId'].nunique()} 部电影")
    log(f"      userId 已重编号为 1 ~ {sub['userId'].max()}（按活跃度降序，1 号最活跃）")

    keep = set(sub["movieId"].unique())
    movies_out = movies[movies["movieId"].isin(keep)].reset_index(drop=True)
    links_out = links[links["movieId"].isin(keep)].reset_index(drop=True)

    # 按用户 85/15 随机划分（等价于 personal_recommender/test.py 的逻辑，但用向量化实现）
    rng = np.random.default_rng(RANDOM_SEED)
    mask = rng.random(len(sub)) < ML_TRAIN_RATIO
    train = sub[mask][["userId", "movieId", "rating"]].copy()
    test = sub[~mask][["userId", "movieId", "rating"]].copy()
    log(f"      划分：train {len(train)} / test {len(test)}（{ML_TRAIN_RATIO:.0%} / {1 - ML_TRAIN_RATIO:.0%}）")

    ratings_out = sub[["userId", "movieId", "rating", "timestamp"]].copy()
    return ratings_out, movies_out, links_out, train, test


# ---------------------------------------------------------------------------
# 5. 附加特征 / 上映国家表
# ---------------------------------------------------------------------------
def build_prediction_extras(train, test):
    log()
    log("[5/8] 构建 data/prediction 增强数据 ...")

    def extras(df):
        return pd.DataFrame({
            "imdb_id": text_col(df["imdb_id"]),
            "popularity2": df["popularity_n"].round(6),
            "rating": df["vote_average_n"].round(6),
            "totalVotes": df["vote_count_n"].round().astype("Int64"),
        })

    train_ex = extras(train)
    test_ex = extras(test)

    # 上映国家表：用 production_countries 重建（详见文件头说明）
    def release_like(df):
        out = pd.DataFrame({
            "original_title": text_col(df["original_title"]),
            "title": text_col(df["title"]),
            "release_year": pd.to_datetime(text_col(df["release_date"]), format="mixed",
                                           errors="coerce").dt.year.astype("Int64"),
            "movie_id": df["id"].astype(int),
        })
        codes = df["production_countries_p"].apply(iso_codes)
        out["theatrical"] = codes.apply(len).astype(int)
        out["theatrical_limited"] = 0
        for cc in COUNTRY_COLS:
            out[cc] = codes.apply(lambda lst, cc=cc: 1 if cc in lst else 0)
        # 行序必须与 train.csv 的 id 顺序完全一致（eda.py 按行号对齐）
        return out.sort_values("movie_id").reset_index(drop=True)

    rel = release_like(train)
    log(f"      TrainAdditionalFeatures {train_ex.shape} / TestAdditionalFeatures {test_ex.shape}")
    log(f"      release_dates_per_country {rel.shape}（theatrical = 制作国家数）")
    return train_ex, test_ex, rel


# ---------------------------------------------------------------------------
# 6. 落盘
# ---------------------------------------------------------------------------
def write_all(train_out, test_out, sub_out, movies, credits,
              ratings_out, movies_out, links_out, ml_train, ml_test,
              train_ex, test_ex, rel):
    log()
    log("[6/8] 写入 data/ 目录 ...")
    os.makedirs(REC_DIR, exist_ok=True)
    os.makedirs(PERSONAL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    train_out.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False, encoding="utf-8")
    test_out.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False, encoding="utf-8")
    sub_out.to_csv(os.path.join(DATA_DIR, "sample_submission.csv"), index=False, encoding="utf-8")

    movies.to_csv(os.path.join(REC_DIR, "tmdb_5000_movies.csv"), index=False, encoding="utf-8")
    credits.to_csv(os.path.join(REC_DIR, "tmdb_5000_credits.csv"), index=False, encoding="utf-8")

    ratings_out.to_csv(os.path.join(PERSONAL_DIR, "ratings.csv"), index=False, encoding="utf-8")
    movies_out.to_csv(os.path.join(PERSONAL_DIR, "movies.csv"), index=False, encoding="utf-8")
    links_out.to_csv(os.path.join(PERSONAL_DIR, "links.csv"), index=False, encoding="utf-8")
    ml_train.to_csv(os.path.join(PERSONAL_DIR, "train.csv"), index=False, encoding="utf-8")
    ml_test.to_csv(os.path.join(PERSONAL_DIR, "test.csv"), index=False, encoding="utf-8")

    train_ex.to_csv(os.path.join(PRED_DIR, "TrainAdditionalFeatures.csv"), index=False, encoding="utf-8")
    test_ex.to_csv(os.path.join(PRED_DIR, "TestAdditionalFeatures.csv"), index=False, encoding="utf-8")
    rel.to_csv(os.path.join(PRED_DIR, "release_dates_per_country.csv"), index=False, encoding="utf-8")

    for p in ("train.csv", "test.csv", "sample_submission.csv",
              "recommendation/tmdb_5000_movies.csv", "recommendation/tmdb_5000_credits.csv",
              "recommendation/personal/ratings.csv", "recommendation/personal/movies.csv",
              "recommendation/personal/links.csv", "recommendation/personal/train.csv",
              "recommendation/personal/test.csv",
              "prediction/TrainAdditionalFeatures.csv", "prediction/TestAdditionalFeatures.csv",
              "prediction/release_dates_per_country.csv"):
        size = os.path.getsize(os.path.join(DATA_DIR, p.replace("/", os.sep)))
        log(f"      {p:<52} {size / 1048576:8.2f} MB")


# ---------------------------------------------------------------------------
# 7. 校验
# ---------------------------------------------------------------------------
LIST_CHECK = ["belongs_to_collection", "genres", "production_companies",
              "production_countries", "spoken_languages", "Keywords", "cast", "crew"]


def verify():
    log()
    log("[7/8] 校验产出（模拟各算法模块的解析方式）...")
    ok = True

    for name in ("train.csv", "test.csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, name))
        need = set(TRAIN_COLS) | ({"revenue"} if name == "train.csv" else set())
        missing = need - set(df.columns)
        if missing:
            log(f"      [FAIL] {name} 缺列 {missing}")
            ok = False
        for c in LIST_CHECK:
            bad = 0
            for v in df[c].head(3000):
                try:
                    ast.literal_eval(v)
                except Exception:
                    bad += 1
            if bad:
                log(f"      [FAIL] {name}.{c} 有 {bad} 行无法 literal_eval")
                ok = False
        log(f"      OK  {name}  shape={df.shape}  列表字段 literal_eval 通过")

    rec = pd.read_csv(os.path.join(REC_DIR, "tmdb_5000_movies.csv"))
    for c in ("id", "title", "overview", "vote_average", "vote_count"):
        if c not in rec.columns:
            log(f"      [FAIL] tmdb_5000_movies.csv 缺列 {c}")
            ok = False
    for c in ("keywords", "genres", "production_companies",
              "production_countries", "spoken_languages"):
        bad = sum(1 for v in rec[c].head(2000) if not _safe_literal(v))
        if bad:
            log(f"      [FAIL] tmdb_5000_movies.csv.{c} 有 {bad} 行无法解析")
            ok = False
    log(f"      OK  tmdb_5000_movies.csv  shape={rec.shape}")

    cr = pd.read_csv(os.path.join(REC_DIR, "tmdb_5000_credits.csv"))
    if list(cr.columns) != ["movie_id", "title", "cast", "crew"]:
        log(f"      [FAIL] tmdb_5000_credits.csv 列名/顺序不对：{list(cr.columns)}")
        ok = False
    else:
        log(f"      OK  tmdb_5000_credits.csv  shape={cr.shape}  列顺序正确")

    for c in ("cast", "crew"):
        bad = sum(1 for v in cr[c].head(2000) if not _safe_literal(v))
        if bad:
            log(f"      [FAIL] tmdb_5000_credits.csv.{c} 有 {bad} 行无法解析")
            ok = False

    pm = pd.read_csv(os.path.join(PERSONAL_DIR, "movies.csv"))
    pr = pd.read_csv(os.path.join(PERSONAL_DIR, "ratings.csv"))
    pl = pd.read_csv(os.path.join(PERSONAL_DIR, "links.csv"))
    for df, cols, nm in ((pm, ["movieId", "title", "genres"], "movies.csv"),
                         (pr, ["userId", "movieId", "rating", "timestamp"], "ratings.csv"),
                         (pl, ["movieId", "imdbId", "tmdbId"], "links.csv")):
        if list(df.columns) != cols:
            log(f"      [FAIL] personal/{nm} 列不对：{list(df.columns)}")
            ok = False
    if list(pd.read_csv(os.path.join(PERSONAL_DIR, "train.csv")).columns) != ["userId", "movieId", "rating"]:
        log("      [FAIL] personal/train.csv 列不对")
        ok = False
    log(f"      OK  personal/  movies={pm.shape} ratings={pr.shape} links={pl.shape}")

    # 关键一致性：release_dates 行序必须与 train.csv 的 id 对齐
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=["id"])
    rel = pd.read_csv(os.path.join(PRED_DIR, "release_dates_per_country.csv"), usecols=["movie_id"])
    if len(rel) != len(tr) or not (rel["movie_id"].values == tr["id"].values).all():
        log("      [FAIL] release_dates_per_country.csv 行序与 train.csv 的 id 不一致")
        ok = False
    else:
        log(f"      OK  release_dates_per_country.csv 行序与 train.csv 对齐（{len(rel)} 行）")

    # tmdbId 能否映射到 tmdb_5000_movies 的 id（影响 ensrable 关键词推荐）
    m_ids = set(rec["id"].astype(int))
    hit = sum(1 for t in pl["tmdbId"].dropna().astype(int) if t in m_ids)
    log(f"      personal links.tmdbId 命中 tmdb_5000_movies 的比例：{hit}/{len(pl)} "
        f"({hit / max(len(pl), 1):.1%})")
    return ok


def _safe_literal(v):
    for parser in (ast.literal_eval, json.loads):
        try:
            parser(v)
            return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
def main():
    log("=" * 78)
    log("数据集替换 / 重建")
    log("=" * 78)

    meta = load_raw()
    train_src, test_src, train_out, test_out, sub_out = build_boxoffice(meta)
    movies, credits = build_recommendation_meta(meta)
    ratings_out, movies_out, links_out, ml_train, ml_test = build_personal()
    train_ex, test_ex, rel = build_prediction_extras(train_src, test_src)

    write_all(train_out, test_out, sub_out, movies, credits,
              ratings_out, movies_out, links_out, ml_train, ml_test,
              train_ex, test_ex, rel)

    ok = verify()
    log()
    log("[8/8] " + ("全部校验通过 ✔" if ok else "存在校验失败项，请查看上面 [FAIL] 行 ✘"))
    log()
    log("下一步：")
    log("    .venv\\Scripts\\python.exe algorithm/FeatureEDA/preprocess.py")
    log("    .venv\\Scripts\\python.exe algorithm/FeatureEDA/eda.py")
    log("    .venv\\Scripts\\python.exe sql/import_data.py --password 123456")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
