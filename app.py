import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="IBD Cluster Notes Demo", layout="wide")
st.title("IBD Cluster Notes Demo")
st.caption("Production-safe mode: cluster summary first, graph only for selected cluster")

# ───────────────────────── Helpers ─────────────────────────

def clean_id(x) -> str:
    s = str(x).strip()
    if s.startswith("[") and "](" in s:
        s = s.split("[", 1)[1].split("]", 1)[0]
    s = re.sub(r"\s+", "", s)
    return s

def classify_relationship(total_cm: float) -> str:
    if total_cm >= 2500:
        return "1st degree"
    elif total_cm >= 1800:
        return "2nd degree"
    elif total_cm >= 1000:
        return "3rd degree"
    return "remote/uncertain"

def make_note_line(row: pd.Series) -> str:
    mt = row.get("haplogroup_mt", "NA")
    y = row.get("haplogroup_y", "NA")
    site = row.get("site", "NA")
    region = row.get("region", "NA")
    return f"{row['sample']} | mt={mt} | Y={y} | site={site} | region={region} | {row['cluster']}"

def detect_separator(sample_bytes: bytes) -> str:
    text = sample_bytes[:20000].decode("utf-8", errors="ignore")
    if text.count("\t") > text.count(",") and text.count("\t") > text.count(";"):
        return "\t"
    if text.count(";") > text.count(","):
        return ";"
    return ","

def norm_float(s) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (float, int)):
        return float(s)
    s = str(s).strip().replace("\u200e", "").replace(" ", "")
    s = s.strip('"').strip("'")
    m = re.search(r"([\d,\.]+)\s*cM", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    s = s.replace("%", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def norm_col(c: str) -> str:
    c = str(c).strip().replace("\ufeff", "")
    c = c.lower()
    c = re.sub(r"[\s\-/]+", "_", c)
    return c

def deduplicate_undirected_pairs(df: pd.DataFrame, keep: str = "max") -> pd.DataFrame:
    x = df.copy()
    x["sample1"] = x["sample1"].apply(clean_id)
    x["sample2"] = x["sample2"].apply(clean_id)
    x["total_cM"] = pd.to_numeric(x["total_cM"], errors="coerce")
    x = x.dropna(subset=["sample1", "sample2", "total_cM"]).copy()
    x = x[x["sample1"] != x["sample2"]].copy()

    pair_sorted = pd.DataFrame(
        np.sort(x[["sample1", "sample2"]].astype(str).values, axis=1),
        columns=["sample1_canon", "sample2_canon"],
        index=x.index,
    )
    x["sample1_canon"] = pair_sorted["sample1_canon"]
    x["sample2_canon"] = pair_sorted["sample2_canon"]

    agg = {"total_cM": "max" if keep == "max" else "sum"}
    for extra in ["platform", "source_file", "relationship_class"]:
        if extra in x.columns:
            agg[extra] = "first"

    out = (
        x.groupby(["sample1_canon", "sample2_canon"], as_index=False)
        .agg(agg)
        .rename(columns={"sample1_canon": "sample1", "sample2_canon": "sample2"})
    )
    return out

# ───────────────────────── ancIBD block TSV parser ─────────────────────────

def parse_ancibd_block_tsv(raw_bytes: bytes, source: str) -> pd.DataFrame:
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    text = raw_bytes.decode("utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    pairs = []
    current_focal = None

    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        if raw_line[:1].isspace():
            if current_focal is None:
                continue
            stripped = line.strip()
            parts = [p for p in stripped.split("\t") if p != ""] if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                continue
            match_id = parts[0].strip()
            cm = None
            for tok in parts[1:]:
                tok2 = tok.strip().replace(",", ".")
                try:
                    cm = float(tok2)
                    break
                except ValueError:
                    continue
            if cm is None:
                continue
            pairs.append({
                "sample1": current_focal,
                "sample2": match_id,
                "total_cM": cm,
                "source_file": source,
                "platform": "ancIBD block TSV",
            })
        else:
            focal = line.strip().split("\t")[0].split()[0]
            if focal:
                current_focal = focal

    return pd.DataFrame(pairs, columns=["sample1", "sample2", "total_cM", "source_file", "platform"])

# ───────────────────────── Multi-CSV parsers ─────────────────────────

def _empty_segs() -> pd.DataFrame:
    return pd.DataFrame(columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"])

def parse_529_segments(df, source):
    df2 = df.rename(columns={
        "Name": "id1", "Match name": "id2", "Chromosome": "chrom",
        "Start point": "start", "End point": "end", "Genetic distance": "cM", "# SNPs": "snps"
    })
    df2["cM"] = df2["cM"].apply(norm_float)
    df2["platform"] = "529andYou"
    df2["source_file"] = source
    pairs = df2.groupby(["id1", "id2"], as_index=False)["cM"].sum().rename(columns={"cM": "total_cM"})
    pairs["platform"] = "529andYou"
    pairs["source_file"] = source
    return pairs, df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]]

def parse_geneanet_segments(df, source):
    df2 = df.rename(columns={
        "Public name": "id2",
        "Username of the member who has uploaded the DNA data": "id1",
        "Chromosome": "chrom", "Start of segment": "start", "Length of segment": "end",
        "Number of SNPs": "snps", "Length in centimorgan (cM)": "cM"
    })
    df2["cM"] = df2["cM"].apply(norm_float)
    df2["platform"] = "Geneanet"
    df2["source_file"] = source
    pairs = df2.groupby(["id1", "id2"], as_index=False)["cM"].sum().rename(columns={"cM": "total_cM"})
    pairs["platform"] = "Geneanet"
    pairs["source_file"] = source
    return pairs, df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]]

def parse_myheritage_matches(df, source, focal):
    cols_map = {c.lower().strip(): c for c in df.columns}
    name_col = cols_map.get("nom") or cols_map.get("name")
    total_cm_col = next((c for c in df.columns if "total de cm" in c.lower() or "total of shared" in c.lower() or "shared dna" in c.lower()), None)
    if name_col is None or total_cm_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()
    df2 = df[[name_col, total_cm_col]].copy()
    df2["total_cM"] = df2[total_cm_col].apply(norm_float)
    df2["id1"] = focal
    df2.rename(columns={name_col: "id2"}, inplace=True)
    df2["platform"] = "MyHeritage"
    df2["source_file"] = source
    return df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"]), _empty_segs()

def parse_myheritage_autocluster(df, source):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_col = cols_lower.get("name")
    cm_col = next((c for c in df.columns if "total" in c.lower() and "cm" in c.lower()), None)
    if name_col is None or cm_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()
    rows = []
    for _, row in df.iterrows():
        cm = norm_float(row[cm_col])
        if cm is None:
            continue
        rows.append({"id1": str(row[name_col]).strip(), "id2": "FOCAL", "total_cM": cm, "platform": "MyHeritage AutoCluster", "source_file": source})
    return pd.DataFrame(rows), _empty_segs()

def parse_ftdna_matches(df, source, focal):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_col = cols_lower.get("full name")
    if name_col is None:
        fn = cols_lower.get("first name")
        ln = cols_lower.get("last name")
        if fn and ln:
            df["_full_name"] = df[fn].astype(str).str.strip() + " " + df[ln].astype(str).str.strip()
            name_col = "_full_name"
    cm_col = cols_lower.get("shared dna")
    if name_col is None or cm_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()
    df2 = df[[name_col, cm_col]].copy()
    df2["total_cM"] = df2[cm_col].apply(norm_float)
    df2["id1"] = focal
    df2.rename(columns={name_col: "id2"}, inplace=True)
    df2["id2"] = df2["id2"].astype(str).str.strip()
    df2["platform"] = "FTDNA"
    df2["source_file"] = source
    return df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"]), _empty_segs()

def parse_23andme_relatives(df, source, focal):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_col = cols_lower.get("display name")
    if name_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()

    if "chromosome number" in cols_lower:
        chrom_col = cols_lower["chromosome number"]
        start_col = cols_lower.get("chromosome start point")
        end_col = cols_lower.get("chromosome end point")
        cm_col = cols_lower.get("genetic distance")
        snp_col = cols_lower.get("# snps")
        df2 = df.copy()
        df2["cM"] = df2[cm_col].apply(norm_float) if cm_col else None
        df2["id1"] = focal
        df2.rename(columns={name_col: "id2"}, inplace=True)
        df2["id2"] = df2["id2"].astype(str).str.strip()
        df2["platform"] = "23andMe"
        df2["source_file"] = source
        df2["chrom"] = df2.get(chrom_col, None)
        df2["start"] = df2.get(start_col, None)
        df2["end"] = df2.get(end_col, None)
        df2["snps"] = df2.get(snp_col, None)
        segs = df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]].dropna(subset=["cM"])
        pairs = df2.groupby(["id1", "id2"], as_index=False)["cM"].sum().rename(columns={"cM": "total_cM"})
        pairs["platform"] = "23andMe"
        pairs["source_file"] = source
        return pairs, segs

    pct_col = cols_lower.get("percent dna shared")
    if pct_col:
        def pct_to_cm(v):
            f = norm_float(str(v).replace("%", ""))
            return round(f * 71, 1) if f is not None else None
        df2 = df[[name_col, pct_col]].copy()
        df2["total_cM"] = df2[pct_col].apply(pct_to_cm)
        df2["id1"] = focal
        df2.rename(columns={name_col: "id2"}, inplace=True)
        df2["id2"] = df2["id2"].astype(str).str.strip()
        df2["platform"] = "23andMe (~)"
        df2["source_file"] = source
        return df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"]), _empty_segs()

    return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()

def detect_and_parse(file, focal_sample):
    raw_bytes = file.getvalue()
    sep = detect_separator(raw_bytes)
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, dtype=str, on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        df = pd.DataFrame()

    cols_set = set(c.lower().strip() for c in df.columns) if not df.empty else set()
    if "name" in cols_set and "match name" in cols_set and "genetic distance" in cols_set:
        p, s = parse_529_segments(df, file.name)
        return p, s, "529andYou (segments)"
    if "public name" in cols_set and any("centimorgan" in c.lower() for c in df.columns):
        p, s = parse_geneanet_segments(df, file.name)
        return p, s, "Geneanet (segments)"
    autocluster_cols = [c for c in df.columns if re.match(r"^\d+_", c.strip())]
    if len(autocluster_cols) >= 5:
        p, s = parse_myheritage_autocluster(df, file.name)
        return p, s, "MyHeritage AutoCluster"
    if any("total de cm" in c.lower() or "total of shared" in c.lower() or "shared dna" in c.lower() for c in df.columns) and any("nom" in c.lower() or "name" in c.lower() for c in df.columns) and "full name" not in cols_set:
        p, s = parse_myheritage_matches(df, file.name, focal_sample)
        return p, s, "MyHeritage (matches)"
    if "shared dna" in cols_set and ("full name" in cols_set or ("first name" in cols_set and "last name" in cols_set)):
        p, s = parse_ftdna_matches(df, file.name, focal_sample)
        return p, s, "FTDNA (matches)"
    if "display name" in cols_set:
        p, s = parse_23andme_relatives(df, file.name, focal_sample)
        return p, s, "23andMe (relatives)"
    return pd.DataFrame(columns=["id1", "id2", "total_cM", "platform", "source_file"]), _empty_segs(), "Unknown/unsupported (yet)"

# ───────────────────────── Cached builders ─────────────────────────

@st.cache_data(show_spinner=False)
def build_pairs_from_ancibd(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    return parse_ancibd_block_tsv(raw_bytes, filename)

@st.cache_data(show_spinner=False)
def build_pairs_from_classic(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    sep = "\t" if filename.lower().endswith(".tsv") else detect_separator(raw_bytes)

    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=sep,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
        header=None,
    )
    first_row = df.iloc[0].astype(str).tolist()
    first_row_norm = [norm_col(c) for c in first_row]
    looks_like_header = any(x in first_row_norm for x in ["id1", "id2", "sample1", "sample2", "total_cm", "ibd_cm", "shared_cm", "cm"])

    if looks_like_header:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=sep,
            dtype=str,
            on_bad_lines="skip",
            low_memory=False,
        )
        df.columns = [str(c).strip() for c in df.columns]
        raw_cols = list(df.columns)

        def find_first(predicates):
            for raw in raw_cols:
                n = norm_col(raw)
                for p in predicates:
                    if p(n):
                        return raw
            return None

        id1_col = find_first([
            lambda n: n == "sample1", lambda n: n == "id1", lambda n: n == "kit1",
            lambda n: n == "profile1", lambda n: n == "person1", lambda n: n == "individual1",
            lambda n: n == "sample_1", lambda n: n == "id_1",
            lambda n: n.endswith("1") and any(k in n for k in ["sample", "id", "kit", "profile", "person", "individual"]),
        ])
        id2_col = find_first([
            lambda n: n == "sample2", lambda n: n == "id2", lambda n: n == "kit2",
            lambda n: n == "profile2", lambda n: n == "person2", lambda n: n == "individual2",
            lambda n: n == "sample_2", lambda n: n == "id_2", lambda n: n == "match",
            lambda n: n == "match_name", lambda n: n == "display_name",
            lambda n: n.endswith("2") and any(k in n for k in ["sample", "id", "kit", "profile", "person", "individual"]),
        ])
        tot_col = find_first([
            lambda n: n == "total_cm", lambda n: n == "shared_cm", lambda n: n == "ibd_cm",
            lambda n: n == "shared_dna", lambda n: n == "genetic_distance", lambda n: n == "length_cm",
            lambda n: n == "tot_cm", lambda n: n == "cm",
            lambda n: "ibd" in n and "cm" in n,
            lambda n: "total" in n and "cm" in n,
            lambda n: "shared" in n and "cm" in n,
            lambda n: "shared" in n and "dna" in n,
            lambda n: "genetic" in n and "distance" in n,
        ])

        if id1_col is None or id2_col is None or tot_col is None:
            raise ValueError(
                "Unsupported classic pairs format. "
                f"Columns found: {raw_cols}. "
                f"Resolved id1={id1_col}, id2={id2_col}, total={tot_col}"
            )

        out = pd.DataFrame({
            "sample1": df[id1_col].apply(clean_id),
            "sample2": df[id2_col].apply(clean_id),
            "total_cM": df[tot_col].apply(norm_float),
        }).dropna(subset=["total_cM"]).copy()
        return out

    if df.shape[1] < 3:
        raise ValueError("Classic pairs file needs at least 3 columns.")

    out = pd.DataFrame({
        "sample1": df.iloc[:, 0].apply(clean_id),
        "sample2": df.iloc[:, 1].apply(clean_id),
        "total_cM": df.iloc[:, 2].apply(norm_float),
    }).dropna(subset=["total_cM"]).copy()
    return out

@st.cache_data(show_spinner=False)
def build_pairs_from_multi(files_payload, focal_sample: str):
    all_pairs, all_segs, summary_rows = [], [], []
    for name, raw_bytes in files_payload:
        f = type("UploadedFileLike", (), {"name": name, "getvalue": lambda self, b=raw_bytes: b})()
        pairs, segs, platform = detect_and_parse(f, focal_sample)
        summary_rows.append({"file": name, "platform_detected": platform, "pairs_rows": len(pairs), "segments_rows": len(segs)})
        if len(pairs):
            all_pairs.append(pairs)
        if len(segs):
            all_segs.append(segs)

    if not all_pairs:
        return None, None, pd.DataFrame(summary_rows)

    df = pd.concat(all_pairs, ignore_index=True)
    df["sample1"] = df["id1"].apply(clean_id)
    df["sample2"] = df["id2"].apply(clean_id)
    df["total_cM"] = pd.to_numeric(df["total_cM"], errors="coerce")
    df = df.dropna(subset=["total_cM"]).copy()
    segs = pd.concat(all_segs, ignore_index=True) if all_segs else None
    return df, segs, pd.DataFrame(summary_rows)

@st.cache_data(show_spinner=False)
def build_graph_objects(df_small: pd.DataFrame):
    G = nx.Graph()
    for _, r in df_small.iterrows():
        G.add_edge(r["sample1"], r["sample2"], weight=float(r["total_cM"]))
    components = list(nx.connected_components(G))
    cluster_map = {}
    cluster_sizes = []
    for i, comp in enumerate(components, start=1):
        cname = f"Cluster {i}"
        comp_list = list(comp)
        for node in comp_list:
            cluster_map[node] = cname
        cluster_sizes.append((cname, len(comp_list)))
    return G, cluster_map, cluster_sizes

@st.cache_data(show_spinner=False)
def build_cluster_summary(df_small: pd.DataFrame, cluster_map: dict):
    tmp = df_small.copy()
    tmp["cluster"] = tmp["sample1"].map(cluster_map)
    summary = tmp.groupby("cluster").agg(
        pair_count=("total_cM", "size"),
        max_cM=("total_cM", "max"),
        mean_cM=("total_cM", "mean"),
    ).reset_index()

    nodes_per_cluster = {}
    for _, row in tmp.iterrows():
        c = row["cluster"]
        nodes_per_cluster.setdefault(c, set()).update([row["sample1"], row["sample2"]])
    summary["node_count"] = summary["cluster"].map(lambda c: len(nodes_per_cluster.get(c, set())))
    return summary.sort_values(["node_count", "pair_count", "max_cM"], ascending=False)

# ───────────────────────── Session state ─────────────────────────

if "favorites" not in st.session_state:
    st.session_state["favorites"] = set()
if "pedigree_notes" not in st.session_state:
    st.session_state["pedigree_notes"] = ""

# ───────────────────────── Sidebar inputs ─────────────────────────

st.sidebar.header("Input mode")
input_mode = st.sidebar.radio("Choose input source", [
    "ancIBD block TSV (no header)",
    "Classic IBD pairs CSV/TSV",
    "Multi-CSV genealogy loader",
])

st.sidebar.header("Optional sample metadata")
st.sidebar.header("Optional sample metadata")
meta_files = st.sidebar.file_uploader(
    "Metadata files (CSV / TSV / AADR .anno / Akbari XLSX)",
    type=["csv", "tsv", "anno", "txt", "xlsx"],
    accept_multiple_files=True,
    key="meta",
)

def norm_meta_col(c: str) -> str:
    c = str(c).strip().replace("\ufeff", "")
    c = re.sub(r"\s+", " ", c)
    return c.lower()

meta = None
if meta_files:
    _meta_frames = []
    for _mf in meta_files:
        _raw = _mf.getvalue()
        _df = load_metadata_file(_raw, _mf.name)
        if not _df.empty:
            _meta_frames.append(_df)
            st.sidebar.caption(f"✓ {_mf.name}: {len(_df):,} rows")
    if _meta_frames:
        meta = merge_meta_frames(_meta_frames)
        st.sidebar.caption(f"Metadata merged: {len(meta):,} total rows")
        _mt_ok = meta["haplogroup_mt"].notna().sum() if "haplogroup_mt" in meta.columns else 0
        _y_ok  = meta["haplogroup_y"].notna().sum()  if "haplogroup_y"  in meta.columns else 0
        _bp_ok = meta["date_mean_bp"].notna().sum()  if "date_mean_bp"  in meta.columns else 0
        _br_ok = meta["broad_region"].notna().sum()  if "broad_region"  in meta.columns else 0
        st.sidebar.caption(f"mt non-null: {_mt_ok:,} | Y non-null: {_y_ok:,}")
        if _bp_ok or _br_ok:
            st.sidebar.caption(f"BP non-null: {_bp_ok:,} | broad_region non-null: {_br_ok:,}")
        st.sidebar.caption(f"BP non-null: {bp_ok:,} | broad_region non-null: {br_ok:,}")

# ───────────────────────── Data loading ─────────────────────────

df = None
segments_df = None
source_label = None

if input_mode == "ancIBD block TSV (no header)":
    anc_file = st.sidebar.file_uploader("Upload ancIBD block file", key="ancibd")
    if anc_file is None:
        st.info("Upload your ancIBD block file. For big files, pre-filtering by cM is strongly recommended.")
        st.stop()
    raw = anc_file.getvalue()
    df = build_pairs_from_ancibd(raw, anc_file.name)
    df = deduplicate_undirected_pairs(df, keep="max")
    st.info(f"Parsed {len(df):,} ancIBD pairs from file after deduplication.")
    if df.empty:
        st.error("No pairs could be extracted. Check the file format.")
        st.stop()
    source_label = f"ancIBD block TSV ({anc_file.name})"
    with st.expander("Parsed pairs preview", expanded=False):
        st.dataframe(df.head(200), width="stretch", height=220)

elif input_mode == "Classic IBD pairs CSV/TSV":
    ibd_file = st.sidebar.file_uploader("Upload classic pairs file", key="ibd")
    if ibd_file is None:
        st.info("Upload a CSV/TSV with id1,id2,total_cM.")
        st.stop()
    raw = ibd_file.getvalue()
    try:
        df = build_pairs_from_classic(raw, ibd_file.name)
        n_before = len(df)
        df = deduplicate_undirected_pairs(df, keep="max")
        st.caption(f"Deduplicated undirected pairs: {n_before:,} -> {len(df):,}")
    except Exception as e:
        st.error(str(e))
        st.stop()
    source_label = f"Classic pairs ({ibd_file.name})"

else:
    focal_sample = st.sidebar.text_input("Focal sample name", value="Encarna Vicente")
    uploaded_files = st.sidebar.file_uploader("Upload one or more genealogy CSV files", type=["csv"], accept_multiple_files=True, key="multi_csv")
    if not uploaded_files:
        st.info("Upload one or more genealogy CSV files to begin.")
        st.stop()
    payload = [(f.name, f.getvalue()) for f in uploaded_files]
    df, segments_df, summary_rows = build_pairs_from_multi(payload, focal_sample)
    st.subheader("Detection summary")
    st.dataframe(summary_rows, width="stretch")
    if df is None or df.empty:
        st.warning("No pairwise matches could be extracted.")
        st.stop()
    n_before = len(df)
    df = deduplicate_undirected_pairs(df, keep="max")
    st.caption(f"Deduplicated undirected pairs: {n_before:,} -> {len(df):,}")
    source_label = "Unified genealogy loader"
    st.download_button(
        "Download unified pairs CSV",
        df[["sample1", "sample2", "total_cM"]].rename(columns={"sample1": "id1", "sample2": "id2"}).to_csv(index=False).encode("utf-8"),
        file_name="unified_pairs.csv", mime="text/csv",
    )

# ───────────────────────── Optional threshold before graph build ─────────────────────────

df["relationship_class"] = df["total_cM"].apply(classify_relationship)

st.subheader(f"Pairwise IBD input ({source_label})")
col_a, col_b = st.columns([3, 1])
with col_a:
    st.dataframe(
        df[["sample1", "sample2", "total_cM", "relationship_class"]].sort_values("total_cM", ascending=False).head(1000),
        width="stretch", height=240,
    )
with col_b:
    cm_min_global = float(df["total_cM"].min())
    cm_max_global = float(df["total_cM"].max())
    if cm_min_global == cm_max_global:
        min_cm_build = cm_min_global
        st.caption(f"All pairs have the same cM value: {cm_min_global:.2f}")
    else:
        min_cm_build = st.number_input(
            "Minimum cM to build clusters",
            min_value=cm_min_global,
            max_value=cm_max_global,
            value=max(40.0, cm_min_global),
            step=1.0,
        )

build_df = df[df["total_cM"] >= min_cm_build].copy()
st.caption(f"Working set after build threshold: {len(build_df):,} pairs out of {len(df):,}")
if build_df.empty:
    st.warning("No pairs remain after the selected minimum cM threshold.")
    st.stop()

with st.spinner("Building graph and cluster summary..."):
    G, cluster_map, cluster_sizes = build_graph_objects(build_df[["sample1", "sample2", "total_cM"]])

meta_lookup = meta.to_dict(orient="index") if meta is not None else {}

rows = []
for node in G.nodes():
    partners = list(G.neighbors(node))
    totals = [G[node][p]["weight"] for p in partners]
    row = {
        "sample": node,
        "cluster": cluster_map[node],
        "partners": len(partners),
        "max_pairwise_cM": max(totals) if totals else 0.0,
    }
    m = meta_lookup.get(clean_id(node))
    if m is not None:
        for key in [
            "haplogroup_mt", "haplogroup_y", "country", "site", "culture", "date_mean_bp",
            "full_date", "location", "region", "broad_region", "Closest cluster", "closest pop",
            "closest sample", "distance", "F_MISS", "N_MISS", "cluster_id", "cluster_size",
            "ID_core", "st1_uid", "date_source",
        ]:
            if key in m:
                row[key] = m.get(key)
    rows.append(row)

df_samples = pd.DataFrame(rows).sort_values(["cluster", "sample"])
cluster_summary = build_cluster_summary(build_df[["sample1", "sample2", "total_cM"]], cluster_map)

st.subheader("Cluster summary")
st.dataframe(cluster_summary.head(500), width="stretch", height=260)

# ───────────────────────── Temporal distribution chart ─────────────────────────

if "date_mean_bp" in df_samples.columns and df_samples["date_mean_bp"].notna().any():
    st.subheader("Temporal distribution by cluster")
    _tmp = df_samples[["cluster", "date_mean_bp"]].copy()
    _tmp["date_mean_bp"] = pd.to_numeric(_tmp["date_mean_bp"], errors="coerce")
    _tmp = _tmp.dropna(subset=["date_mean_bp"])

    if not _tmp.empty:
        # Limitem als 20 clusters amb més mostres datades
        _top_clusters = (
            _tmp.groupby("cluster")["date_mean_bp"].count()
            .sort_values(ascending=False)
            .head(20).index.tolist()
        )
        _tmp2 = _tmp[_tmp["cluster"].isin(_top_clusters)]

        import plotly.express as plx
        _color_seq = plx.colors.qualitative.Plotly
        _cluster_colors = {c: _color_seq[i % len(_color_seq)] for i, c in enumerate(_top_clusters)}

        fig_time = go.Figure()
        for cl in _top_clusters:
            sub = _tmp2[_tmp2["cluster"] == cl]["date_mean_bp"]
            if sub.empty:
                continue
            fig_time.add_trace(go.Violin(
                x=sub,
                name=cl,
                orientation="h",
                side="positive",
                width=1.8,
                line_color=_cluster_colors[cl],
                meanline_visible=True,
                points="outliers",
                showlegend=True,
            ))
        fig_time.update_layout(
            title="Date mean BP distribution — top 20 clusters",
            xaxis_title="Date mean (years BP, older → left)",
            xaxis=dict(autorange="reversed"),
            yaxis_title="Cluster",
            height=max(350, len(_top_clusters) * 32),
            margin=dict(l=10, r=10, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_time, use_container_width=True)


if meta is not None and not df_samples.empty:
    matched = df_samples["sample"].astype(str).apply(clean_id).isin(meta.index).sum()
    st.caption(f"Metadata matched to {matched} / {len(df_samples)} samples in current graph")

# ───────────────────────── Sidebar cluster controls ─────────────────────────

st.sidebar.header("Cluster selection")
clusters = cluster_summary["cluster"].tolist()

search_id = st.sidebar.text_input("Search sample ID (partial ok)")
cluster_from_id = None
if search_id and not df_samples.empty:
    hits = df_samples[df_samples["sample"].astype(str).str.contains(search_id.strip(), case=False, na=False, regex=False)]
    hit_clusters = sorted(hits["cluster"].unique())
    if len(hit_clusters) == 1:
        cluster_from_id = hit_clusters[0]
    elif len(hit_clusters) > 1:
        cluster_from_id = st.sidebar.selectbox("Multiple clusters matched this ID", hit_clusters)
    st.sidebar.write(f"{len(hits)} samples matched." if len(hits) else "No samples matched.")

cluster_from_hg = None
if meta is not None and not df_samples.empty:
    search_haplo = st.sidebar.text_input("Search haplogroup (mt or Y)")
    if search_haplo:
        hg_cols = [c for c in df_samples.columns if "haplogroup" in c.lower()]
        if hg_cols:
            mask = pd.Series(False, index=df_samples.index)
            for c in hg_cols:
                mask = mask | df_samples[c].astype(str).str.contains(search_haplo.strip(), case=False, na=False, regex=False)
            hits_hg = df_samples[mask]
            hit_clusters_hg = sorted(hits_hg["cluster"].unique())
            n_hits = len(hits_hg)
            n_clusters = len(hit_clusters_hg)

            if n_hits == 0:
                st.sidebar.warning("No haplogroups matched.")
            else:
                st.sidebar.caption(f"{n_hits} samples with '{search_haplo}' in {n_clusters} cluster{'s' if n_clusters != 1 else ''}.")
                summary_hg = (
                    hits_hg.groupby("cluster")
                    .agg(
                        n_samples=("sample", "count"),
                        samples=("sample", lambda x: ", ".join(list(map(str, x))[:5]) + ("..." if len(x) > 5 else "")),
                    )
                    .reset_index()
                    .rename(columns={"cluster": "Cluster", "n_samples": "N samples", "samples": "Examples"})
                )
                st.sidebar.dataframe(summary_hg, height=min(240, 35 * (n_clusters + 1)), use_container_width=True)
                if n_clusters == 1:
                    cluster_from_hg = hit_clusters_hg[0]
                else:
                    cluster_from_hg = st.sidebar.selectbox(
                        f"Select cluster to inspect ({n_clusters} found)",
                        hit_clusters_hg,
                        key="hg_cluster_select",
                    )

default_cluster = cluster_from_id or cluster_from_hg or (clusters[0] if clusters else None)
if default_cluster is None:
    st.warning("No clusters available after filtering.")
    st.stop()

selected = st.sidebar.selectbox("Select cluster to inspect", clusters, index=clusters.index(default_cluster) if default_cluster in clusters else 0)

cf1, cf2 = st.sidebar.columns(2)
with cf1:
    if st.button("Favorite"):
        st.session_state["favorites"].add(selected)
with cf2:
    if st.button("Clear favs"):
        st.session_state["favorites"] = set()
if st.session_state["favorites"]:
    st.sidebar.markdown("**Favorites**")
    for c in sorted(st.session_state["favorites"]):
        st.sidebar.write(f"- {c}")


# ───────────────────────── Sidebar: Geography & Culture filters ─────────────────────────

st.sidebar.header("Geography & Culture filters")

_br_vals = sorted(df_samples["broad_region"].dropna().unique().tolist()) if "broad_region" in df_samples.columns else []
_cult_vals = sorted(df_samples["culture"].dropna().unique().tolist()) if "culture" in df_samples.columns else []
_country_vals = sorted(df_samples["country"].dropna().unique().tolist()) if "country" in df_samples.columns else []

filter_br = []
filter_cult = []
filter_country = []

if _br_vals:
    filter_br = st.sidebar.multiselect("Broad region", _br_vals, default=[], key="f_broad_region")
if _cult_vals:
    filter_cult = st.sidebar.multiselect("Culture", _cult_vals, default=[], key="f_culture")
if _country_vals:
    filter_country = st.sidebar.multiselect("Country", _country_vals, default=[], key="f_country")

def apply_geo_filters(df_s: pd.DataFrame) -> pd.DataFrame:
    out = df_s.copy()
    if filter_br and "broad_region" in out.columns:
        out = out[out["broad_region"].isin(filter_br)]
    if filter_cult and "culture" in out.columns:
        out = out[out["culture"].isin(filter_cult)]
    if filter_country and "country" in out.columns:
        out = out[out["country"].isin(filter_country)]
    return out

# ───────────────────────── Selected cluster only ─────────────────────────

selected_nodes = [n for n, c in cluster_map.items() if c == selected]
selected_pairs = build_df[build_df["sample1"].isin(selected_nodes) & build_df["sample2"].isin(selected_nodes)].copy()
selected_samples = apply_geo_filters(df_samples[df_samples["cluster"] == selected].copy())
if filter_br or filter_cult or filter_country:
    st.caption(f"⚑ Geo filter active — {len(selected_samples)} of {(df_samples['cluster'] == selected).sum()} samples shown")

left, right = st.columns([1.05, 1.15])
with left:
    st.subheader(f"Samples in {selected}")
    st.dataframe(selected_samples.head(1000), width="stretch", height=320)
    # Export CSV del cluster
    _csv_bytes = selected_samples.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download cluster CSV",
        data=_csv_bytes,
        file_name=f"{selected.replace(' ', '_')}_samples.csv",
        mime="text/csv",
        key="dl_cluster_csv",
    )

    st.subheader("Pedigree notes console")
    sample_choices = selected_samples["sample"].tolist()
    sel_note = st.selectbox("Select sample to append", sample_choices if sample_choices else [""], key="sample_to_add_notes")
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("Add sample") and sample_choices:
            row = selected_samples[selected_samples["sample"] == sel_note].iloc[0]
            line = make_note_line(row)
            if line not in st.session_state["pedigree_notes"]:
                st.session_state["pedigree_notes"] += line + "\n"
    with n2:
        if st.button("Add cluster") and not selected_samples.empty:
            existing = set(filter(None, st.session_state["pedigree_notes"].splitlines()))
            for _, r in selected_samples.iterrows():
                line = make_note_line(r)
                if line not in existing:
                    st.session_state["pedigree_notes"] += line + "\n"
                    existing.add(line)
    with n3:
        if st.button("Clear notes"):
            st.session_state["pedigree_notes"] = ""
    st.text_area("Notes", key="pedigree_notes", height=180)
    st.download_button("Download notes.txt", st.session_state["pedigree_notes"].encode("utf-8"), file_name="pedigree_notes.txt", mime="text/plain")

with right:
    st.subheader(f"Pairwise relationships in {selected}")
    if selected_pairs.empty:
        st.info("No pairwise relationships remain in this cluster after filtering.")
    else:
        min_c = float(selected_pairs["total_cM"].min())
        max_c = float(selected_pairs["total_cM"].max())

        if min_c == max_c:
            st.caption(f"Single pair in cluster: {min_c:.2f} cM -- no filter applied.")
            min_cm = min_c
        else:
            fc1, fc2 = st.columns([3, 1])
            with fc1:
                min_cm_view = st.slider(
                    "Min total IBD (cM) in selected cluster",
                    min_value=min_c,
                    max_value=max_c,
                    value=min_c,
                    step=1.0,
                )
            with fc2:
                min_cm_exact = st.number_input(
                    "Exact cM",
                    min_value=min_c,
                    max_value=max_c,
                    value=min_cm_view,
                    step=1.0,
                )
            min_cm = min_cm_exact if min_cm_exact != min_cm_view else min_cm_view

        view_pairs = selected_pairs[selected_pairs["total_cM"] >= min_cm].copy()
        st.dataframe(view_pairs[["sample1", "sample2", "total_cM"]].sort_values("total_cM", ascending=False).head(1000), width="stretch", height=220)

        if view_pairs.empty:
            st.warning("No edges remain in this cluster view after the selected cM threshold.")
        else:
            H = nx.Graph()
            for _, r in view_pairs.iterrows():
                H.add_edge(r["sample1"], r["sample2"], weight=float(r["total_cM"]))

            nodes = list(H.nodes())
            if len(nodes) > 300:
                st.warning(f"Graph limited to top 300 nodes in this cluster (total: {len(nodes)}).")
                degree_rank = sorted(H.degree, key=lambda x: x[1], reverse=True)[:300]
                keep = [n for n, _ in degree_rank]
                H = H.subgraph(keep).copy()

            try:
                pos = nx.spring_layout(H, weight="weight", seed=42)
            except Exception:
                pos = nx.circular_layout(H)


            node_x, node_y, text_labels, hovertext, node_colors = [], [], [], [], []
            for n, (px, py) in pos.items():
                node_x.append(px)
                node_y.append(py)
                text_labels.append(n)
                label_parts = [str(n)]
                m = meta_lookup.get(clean_id(n))
                if m is not None:
                    for k, short in [
                        ("haplogroup_mt", "mt"), ("haplogroup_y", "Y"), ("site", "site"),
                        ("culture", "culture"), ("region", "region"), ("full_date", "date"),
                        ("date_mean_bp", "BP"), ("broad_region", "geo"),
                        ("Closest cluster", "closest_cluster"), ("closest pop", "closest_pop"),
                        ("distance", "dist"), ("F_MISS", "F_MISS"),
                    ]:
                        if k in m and m.get(k) is not None and pd.notna(m.get(k)):
                            label_parts.append(f"{short}: {m.get(k)}")
                hovertext.append(" | ".join(label_parts))
                node_colors.append(m.get("broad_region", "Other") if m is not None else "Other")

            edge_x, edge_y = [], []
            for u, v in H.edges():
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=1), hoverinfo="none"))
            # Coloració per broad_region
            _unique_regions = list(dict.fromkeys(node_colors))
            _cmap = {}
            _palette = [
                "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
                "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
                "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
            ]
            for _ri, _rv in enumerate(_unique_regions):
                _cmap[_rv] = _palette[_ri % len(_palette)]
            _node_color_vals = [_cmap.get(c, "#888888") for c in node_colors]
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=text_labels,
                textposition="top center",
                marker=dict(
                    size=11,
                    color=_node_color_vals,
                    line=dict(width=0.8, color="white"),
                ),
                hovertext=hovertext,
                hoverinfo="text",
                customdata=node_colors,
                name="samples",
            ))
            # Llegenda de regions
            for _rv, _rc in _cmap.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=_rc),
                    name=str(_rv),
                    showlegend=True,
                ))
            fig.update_layout(title=f"IBD network — {selected}", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=True, legend=dict(title="Broad region", itemsizing="constant", orientation="v", x=1.01, y=1), height=700, margin=dict(l=10, r=160, t=50, b=10))
            st.plotly_chart(fig, width="stretch")

if segments_df is not None:
    st.subheader("Unified segments preview")
    st.dataframe(segments_df.head(200), width="stretch", height=200)
