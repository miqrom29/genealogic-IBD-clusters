import io
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Universal Genealogy Loader", layout="wide")

st.title("Universal Genealogy DNA Match Loader (beta)")

st.markdown(
    "Upload one or more CSV files from MyHeritage, Geneanet, 529andYou, "
    "23andMe or FTDNA. The app will try to detect the format and "
    "normalize everything to a common model."
)

# ───────────────────────── Helpers ─────────────────────────


def detect_separator(sample_bytes: bytes) -> str:
    # Very simple heuristic: if there are many ';', assume semicolon
    text = sample_bytes.decode("utf-8", errors="ignore")
    if text.count(";") > text.count(","):
        return ";"
    return ","


def norm_float(s: str) -> Optional[float]:
    """
    Normalize European decimals like '13,4' to float 13.4.
    """
    if s is None:
        return None
    if isinstance(s, float) or isinstance(s, int):
        return float(s)
    s = str(s).strip().replace("‎", "")
    s = s.replace(" ", "")
    # remove quotes if any
    s = s.strip('"').strip("'")
    # handle values like '13,4' or '13.4'
    if "," in s and "." in s:
        # if both present, try last separator as decimal mark
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        # handle patterns like "0.1% (8.6 cM)"
        m = re.search(r"\(([0-9]+(?:\.[0-9]+)?)\s*cM", s)
        if m:
            return float(m.group(1))
    return None


# ───────────────────────── Parsers ─────────────────────────


def parse_529_segments(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Name,Match name,Chromosome,Start point,End point,Genetic distance,# SNPs
    df2 = df.rename(
        columns={
            "Name": "id1",
            "Match name": "id2",
            "Chromosome": "chrom",
            "Start point": "start",
            "End point": "end",
            "Genetic distance": "cM",
            "# SNPs": "snps",
        }
    )
    df2["cM"] = df2["cM"].apply(norm_float)
    df2["platform"] = "529andYou"
    df2["source_file"] = source

    # We only have segments; to get total cM per pair we sum
    pairs = (
        df2.groupby(["id1", "id2"], as_index=False)["cM"]
        .sum()
        .rename(columns={"cM": "total_cM"})
    )
    pairs["platform"] = "529andYou"
    pairs["source_file"] = source

    segs = df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]]

    return pairs, segs


def parse_geneanet_segments(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Public name,Username...,Chromosome,Start of segment,Length of segment,Number of SNPs,Length in centimorgan (cM),Type of segment
    df2 = df.rename(
        columns={
            "Public name": "id2",
            "Username of the member who has uploaded the DNA data": "id1",
            "Chromosome": "chrom",
            "Start of segment": "start",
            "Length of segment": "end",
            "Number of SNPs": "snps",
            "Length in centimorgan (cM)": "cM",
            "Type of segment": "segment_type",
        }
    )
    df2["cM"] = df2["cM"].apply(norm_float)
    df2["platform"] = "Geneanet"
    df2["source_file"] = source

    pairs = (
        df2.groupby(["id1", "id2"], as_index=False)["cM"]
        .sum()
        .rename(columns={"cM": "total_cM"})
    )
    pairs["platform"] = "Geneanet"
    pairs["source_file"] = source

    segs = df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]]

    return pairs, segs


def parse_myheritage_matches(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ID de Coincidència d'ADN, Nom, ..., Total de cM compartits, Percentatge d'ADN compartit, ...
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("nom") or cols.get("name")
    total_cm_col = None
    for c in df.columns:
        if "Total de cM" in c or "Total cM" in c or "Total of shared DNA" in c:
            total_cm_col = c
            break
    if name_col is None or total_cm_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), pd.DataFrame(
            columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]
        )

    focal = "Encarna Vicente"  # podríem fer-ho parametritzable després
    df2 = df[[name_col, total_cm_col]].copy()
    df2["total_cM"] = df2[total_cm_col].apply(norm_float)
    df2["id1"] = focal
    df2.rename(columns={name_col: "id2"}, inplace=True)
    df2["platform"] = "MyHeritage"
    df2["source_file"] = source

    pairs = df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"])

    segs = pd.DataFrame(
        columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]
    )
    return pairs, segs


def detect_and_parse(file) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    raw_bytes = file.getvalue()
    sep = detect_separator(raw_bytes)
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, dtype=str)

    cols_lower = [c.lower() for c in df.columns]

    # 529andYou segments
    if "name" in cols_lower and "match name" in cols_lower and "genetic distance" in cols_lower:
        pairs, segs = parse_529_segments(df, file.name)
        platform = "529andYou (segments)"
    # Geneanet segments
    elif "public name" in cols_lower and "length in centimorgan (cm)" in cols_lower:
        pairs, segs = parse_geneanet_segments(df, file.name)
        platform = "Geneanet (segments)"
    # MyHeritage matches
    elif any("total de cm compartits" in c.lower() for c in df.columns):
        pairs, segs = parse_myheritage_matches(df, file.name)
        platform = "MyHeritage (matches)"
    else:
        # Placeholder for other formats (FTDNA, 23andMe, etc.)
        platform = "Unknown/unsupported (yet)"
        pairs = pd.DataFrame(columns=["id1", "id2", "total_cM", "platform", "source_file"])
        segs = pd.DataFrame(
            columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]
        )

    return pairs, segs, platform


# ───────────────────────── UI ─────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or more CSV files from DNA platforms",
    type=["csv"],
    accept_multiple_files=True,
)

all_pairs: List[pd.DataFrame] = []
all_segs: List[pd.DataFrame] = []

if uploaded_files:
    summary_rows = []
    for f in uploaded_files:
        pairs, segs, platform = detect_and_parse(f)
        summary_rows.append(
            {
                "file": f.name,
                "platform_detected": platform,
                "pairs_rows": len(pairs),
                "segments_rows": len(segs),
            }
        )
        if len(pairs):
            all_pairs.append(pairs)
        if len(segs):
            all_segs.append(segs)

    st.subheader("Detection summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if all_pairs:
        pairs_df = pd.concat(all_pairs, ignore_index=True)
        st.subheader("Unified pairwise matches (id1,id2,total_cM)")
        st.dataframe(pairs_df.head(200), use_container_width=True)
        csv_bytes = pairs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download unified pairs CSV",
            csv_bytes,
            file_name="unified_pairs_for_ibd_app.csv",
            mime="text/csv",
        )
    else:
        pairs_df = None
        st.info("No pairwise matches could be extracted yet.")

    if all_segs:
        segs_df = pd.concat(all_segs, ignore_index=True)
        st.subheader("Unified segments (optional)")
        st.dataframe(segs_df.head(200), use_container_width=True)
        csv_bytes2 = segs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download unified segments CSV",
            csv_bytes2,
            file_name="unified_segments_for_ibd_app.csv",
            mime="text/csv",
        )
    else:
        segs_df = None
        st.info("No segment-level data extracted yet.")
else:
    st.info("Upload at least one CSV file to start.")
