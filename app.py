import io
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="IBD Cluster Notes Demo", layout="wide")
st.title("IBD Cluster Notes Demo")
st.caption("Classic IBD cluster mode + multi-CSV genealogy loader + pedigree notes console")

# ───────────────────────── Shared helpers ─────────────────────────

def clean_id(x: str) -> str:
    s = str(x)
    if s.startswith("[") and "](" in s:
        return s.split("[", 1)[1].split("]", 1)[0]
    return s.strip()


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
    return f"{row['sample']} | mt={mt} | Y={y} | {row['cluster']}"


def detect_separator(sample_bytes: bytes) -> str:
    text = sample_bytes.decode("utf-8", errors="ignore")
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
    # "21.44 cM" pattern
    m = re.search(r"([\d,\.]+)\s*cM", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    # "0.16%" -> percent to cM rough skip, keep raw number
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

# ───────────────────────── Multi-loader parsers ─────────────────────────

def parse_529_segments(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.rename(columns={
        "Name": "id1", "Match name": "id2", "Chromosome": "chrom",
        "Start point": "start", "End point": "end",
        "Genetic distance": "cM", "# SNPs": "snps",
    })
    df2["cM"] = df2["cM"].apply(norm_float)
    df2["platform"] = "529andYou"
    df2["source_file"] = source
    pairs = df2.groupby(["id1", "id2"], as_index=False)["cM"].sum().rename(columns={"cM": "total_cM"})
    pairs["platform"] = "529andYou"
    pairs["source_file"] = source
    segs = df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]]
    return pairs, segs


def parse_geneanet_segments(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.rename(columns={
        "Public name": "id2",
        "Username of the member who has uploaded the DNA data": "id1",
        "Chromosome": "chrom", "Start of segment": "start",
        "Length of segment": "end", "Number of SNPs": "snps",
        "Length in centimorgan (cM)": "cM", "Type of segment": "segment_type",
    })
    df2["cM"] = df2["cM"].apply(norm_float)
    df2["platform"] = "Geneanet"
    df2["source_file"] = source
    pairs = df2.groupby(["id1", "id2"], as_index=False)["cM"].sum().rename(columns={"cM": "total_cM"})
    pairs["platform"] = "Geneanet"
    pairs["source_file"] = source
    segs = df2[["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"]]
    return pairs, segs


def parse_myheritage_matches(df: pd.DataFrame, source: str, focal: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Standard MH match list
    cols_map = {c.lower().strip(): c for c in df.columns}
    name_col = cols_map.get("nom") or cols_map.get("name")
    total_cm_col = None
    for c in df.columns:
        cl = c.lower()
        if "total de cm compartits" in cl or "total of shared dna" in cl or "total shared cm" in cl or "shared dna" in cl:
            total_cm_col = c
            break
    if name_col is None or total_cm_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()
    df2 = df[[name_col, total_cm_col]].copy()
    df2["total_cM"] = df2[total_cm_col].apply(norm_float)
    df2["id1"] = focal
    df2.rename(columns={name_col: "id2"}, inplace=True)
    df2["platform"] = "MyHeritage"
    df2["source_file"] = source
    pairs = df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"])
    return pairs, _empty_segs()


def parse_myheritage_autocluster(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    MH Auto Cluster format:
    Identifier | Name | total shared cM | Tree | Tree Person Count | Notes | Cluster | 1_Name | 2_Name ...
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_col = cols_lower.get("name")
    cm_col = None
    for c in df.columns:
        if "total" in c.lower() and "cm" in c.lower():
            cm_col = c
            break
    if name_col is None or cm_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()

    # Cluster membership columns start with a number and underscore
    cluster_cols = [c for c in df.columns if re.match(r"^\d+_", c.strip())]

    pairs_list = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        cm = norm_float(row[cm_col])
        if cm is None:
            continue
        # Find which cluster columns have a truthy value
        member_clusters = []
        for cc in cluster_cols:
            val = str(row.get(cc, "")).strip()
            if val not in ("", "0", "nan", "None"):
                member_clusters.append(cc)
        # Pair: each cluster member with the focal person (name) using total cM
        # We don't have pairwise cMs within cluster, so we store focal–match pairs
        # using shared cM as the edge weight (same as classic MH match list)
        pairs_list.append({"id1": name, "id2": "FOCAL", "total_cM": cm,
                            "platform": "MyHeritage AutoCluster", "source_file": source})

    pairs = pd.DataFrame(pairs_list)
    return pairs, _empty_segs()


def parse_ftdna_matches(df: pd.DataFrame, source: str, focal: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    FTDNA Family Finder matches:
    Full Name | First Name | ... | Shared DNA | Longest Block | Y-DNA Haplogroup | mtDNA Haplogroup ...
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    # Name: prefer "Full Name", fallback "First Name" + "Last Name"
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
    pairs = df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"])
    return pairs, _empty_segs()


def parse_23andme_relatives(df: pd.DataFrame, source: str, focal: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    23andMe relatives export (full, with haplogroups):
    Display Name | ... | Percent DNA Shared | # Segments Shared | Maternal Haplogroup | Paternal Haplogroup ...
    Also handles segment-level rows: Display Name | Chromosome Number | Start | End | Genetic Distance | # SNPs
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_col = cols_lower.get("display name")
    if name_col is None:
        return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()

    # Segment-level data? Has chromosome number col
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

    # Match-level data: uses "Percent DNA Shared"
    pct_col = cols_lower.get("percent dna shared")
    if pct_col:
        df2 = df[[name_col, pct_col]].copy()
        # Convert % to rough cM: 1% ≈ 71 cM
        def pct_to_cm(v):
            f = norm_float(str(v).replace("%", ""))
            return round(f * 71, 1) if f is not None else None
        df2["total_cM"] = df2[pct_col].apply(pct_to_cm)
        df2["id1"] = focal
        df2.rename(columns={name_col: "id2"}, inplace=True)
        df2["id2"] = df2["id2"].astype(str).str.strip()
        df2["platform"] = "23andMe (~)"
        df2["source_file"] = source
        pairs = df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"])
        return pairs, _empty_segs()

    return pd.DataFrame(columns=["id1", "id2", "total_cM"]), _empty_segs()


def _empty_segs() -> pd.DataFrame:
    return pd.DataFrame(columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"])


def detect_and_parse(file, focal_sample: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    raw_bytes = file.getvalue()
    sep = detect_separator(raw_bytes)
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, dtype=str, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, dtype=str, error_bad_lines=False)

    cols_set = set(c.lower().strip() for c in df.columns)

    # 529andYou segments
    if "name" in cols_set and "match name" in cols_set and "genetic distance" in cols_set:
        p, s = parse_529_segments(df, file.name)
        return p, s, "529andYou (segments)"

    # Geneanet segments
    if "public name" in cols_set and any("centimorgan" in c.lower() for c in df.columns):
        p, s = parse_geneanet_segments(df, file.name)
        return p, s, "Geneanet (segments)"

    # MyHeritage AutoCluster (has columns like "1_Name", many of them)
    autocluster_cols = [c for c in df.columns if re.match(r"^\d+_", c.strip())]
    if len(autocluster_cols) >= 5:
        p, s = parse_myheritage_autocluster(df, file.name)
        return p, s, "MyHeritage AutoCluster"

    # MyHeritage standard match list
    if any("total de cm compartits" in c.lower() or "total of shared dna" in c.lower() or "shared dna" in c.lower() for c in df.columns) and \
       any("nom" in c.lower() or "name" in c.lower() for c in df.columns) and \
       "full name" not in cols_set:
        p, s = parse_myheritage_matches(df, file.name, focal_sample)
        return p, s, "MyHeritage (matches)"

    # FTDNA matches (Full Name or First Name + Shared DNA columns)
    if ("shared dna" in cols_set) and ("full name" in cols_set or ("first name" in cols_set and "last name" in cols_set)):
        p, s = parse_ftdna_matches(df, file.name, focal_sample)
        return p, s, "FTDNA (matches)"

    # 23andMe relatives (Display Name col)
    if "display name" in cols_set:
        p, s = parse_23andme_relatives(df, file.name, focal_sample)
        return p, s, "23andMe (relatives)"

    return pd.DataFrame(columns=["id1", "id2", "total_cM", "platform", "source_file"]), _empty_segs(), "Unknown/unsupported (yet)"

# ───────────────────────── Session state ─────────────────────────

if "favorites" not in st.session_state:
    st.session_state["favorites"] = set()
if "pedigree_notes" not in st.session_state:
    st.session_state["pedigree_notes"] = ""

# ───────────────────────── Sidebar: mode & metadata ─────────────────────────

st.sidebar.header("Input mode")
input_mode = st.sidebar.radio("Choose input source", [
    "Classic IBD pairs file",
    "Multi-CSV genealogy loader",
])

st.sidebar.header("Optional sample metadata")
meta_file = st.sidebar.file_uploader(
    "Metadata CSV (sample,haplogroup_mt,mt_haplogroup,y_haplogroup,...)",
    type=["csv"], key="meta",
)

meta = None
if meta_file is not None:
    meta = pd.read_csv(meta_file)
    meta_cols = {c.lower(): c for c in meta.columns}
    sid_col = meta_cols.get("sample") or list(meta.columns)[0]
    meta["sample_clean"] = meta[sid_col].apply(clean_id)
    meta = meta.set_index("sample_clean")
    if "haplogroup_mt" not in meta.columns:
        col_mt = meta_cols.get("mt_haplogroup")
        if col_mt:
            meta.rename(columns={col_mt: "haplogroup_mt"}, inplace=True)
    if "haplogroup_y" not in meta.columns:
        col_y = meta_cols.get("y_haplogroup")
        if col_y:
            meta.rename(columns={col_y: "haplogroup_y"}, inplace=True)

# ───────────────────────── Build pair table ─────────────────────────

df = None
segments_df = None

if input_mode == "Classic IBD pairs file":
    st.sidebar.header("Upload IBD pairs")
    ibd_file = st.sidebar.file_uploader("CSV or TSV with id1,id2,total_cM", type=["csv", "tsv"], key="ibd")
    if ibd_file is None:
        st.info("Upload a CSV/TSV file with id1,id2,total_cM to begin.")
        st.stop()
    sep = "\t" if ibd_file.name.endswith(".tsv") else ","
    df = pd.read_csv(ibd_file, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    id1_col = cols.get("sample1") or cols.get("id1") or list(df.columns)[0]
    id2_col = cols.get("sample2") or cols.get("id2") or list(df.columns)[1]
    tot_col = cols.get("total_cm") or cols.get("length_cm") or cols.get("tot_cm") or list(df.columns)[2]
    df["sample1"] = df[id1_col].apply(clean_id)
    df["sample2"] = df[id2_col].apply(clean_id)
    df["total_cM"] = pd.to_numeric(df[tot_col], errors="coerce")
    df = df.dropna(subset=["total_cM"]).copy()
    source_label = "Classic IBD pairs"

else:
    st.sidebar.header("Upload genealogy CSVs")
    focal_sample = st.sidebar.text_input("Focal sample name (for match-only files)", value="Encarna Vicente")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV files from DNA platforms",
        type=["csv"], accept_multiple_files=True, key="multi_csv",
    )
    if not uploaded_files:
        st.info("Upload one or more genealogy CSV files (FTDNA, MyHeritage, 23andMe, Geneanet, 529andYou) to begin.")
        st.stop()
    all_pairs: List[pd.DataFrame] = []
    all_segs: List[pd.DataFrame] = []
    summary_rows = []
    for f in uploaded_files:
        pairs, segs, platform = detect_and_parse(f, focal_sample)
        summary_rows.append({
            "file": f.name,
            "platform_detected": platform,
            "pairs_rows": len(pairs),
            "segments_rows": len(segs),
        })
        if len(pairs):
            all_pairs.append(pairs)
        if len(segs):
            all_segs.append(segs)
    st.subheader("Detection summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    if not all_pairs:
        st.warning("No pairwise matches could be extracted from the uploaded files.")
        st.stop()
    df = pd.concat(all_pairs, ignore_index=True)
    df["sample1"] = df["id1"].apply(clean_id)
    df["sample2"] = df["id2"].apply(clean_id)
    df["total_cM"] = pd.to_numeric(df["total_cM"], errors="coerce")
    df = df.dropna(subset=["total_cM"]).copy()
    if all_segs:
        segments_df = pd.concat(all_segs, ignore_index=True)
    source_label = "Unified genealogy loader"
    st.download_button(
        "Download unified pairs CSV",
        df[["sample1", "sample2", "total_cM"]].rename(columns={"sample1": "id1", "sample2": "id2"}).to_csv(index=False).encode("utf-8"),
        file_name="unified_pairs_for_ibd_app.csv", mime="text/csv",
    )
    if segments_df is not None:
        st.download_button(
            "Download unified segments CSV",
            segments_df.to_csv(index=False).encode("utf-8"),
            file_name="unified_segments_for_ibd_app.csv", mime="text/csv",
        )

df["relationship_class"] = df["total_cM"].apply(classify_relationship)

st.subheader(f"Pairwise IBD input ({source_label})")
st.dataframe(
    df[["sample1", "sample2", "total_cM", "relationship_class"]].sort_values("total_cM", ascending=False),
    use_container_width=True, height=240,
)

G = nx.Graph()
for _, r in df.iterrows():
    G.add_edge(r["sample1"], r["sample2"], weight=r["total_cM"])

components = list(nx.connected_components(G))
cluster_map = {}
for i, comp in enumerate(components, start=1):
    label = f"Cluster {i}"
    for node in comp:
        cluster_map[node] = label

rows = []
for node in G.nodes():
    partners = list(G.neighbors(node))
    totals = [G[node][p]["weight"] for p in partners]
    row = {"sample": node, "cluster": cluster_map[node], "partners": len(partners),
           "max_pairwise_cM": max(totals) if totals else 0.0}
    if meta is not None and node in meta.index:
        for col in meta.columns:
            row[col] = meta.loc[node, col]
    rows.append(row)

df_samples = pd.DataFrame(rows).sort_values(["cluster", "sample"])
st.subheader("Sample assignments")
st.dataframe(df_samples, use_container_width=True, height=240)

# ───────────────────────── Sidebar: cluster search ─────────────────────────

st.sidebar.header("Cluster view")
clusters = sorted(df_samples["cluster"].unique())
search_id = st.sidebar.text_input("Search sample ID (partial ok)")
cluster_from_id = None
if search_id:
    hits = df_samples[df_samples["sample"].astype(str).str.contains(search_id.strip(), case=False, na=False, regex=False)]
    hit_clusters = sorted(hits["cluster"].unique())
    if len(hit_clusters) == 1:
        cluster_from_id = hit_clusters[0]
    elif len(hit_clusters) > 1:
        cluster_from_id = st.sidebar.selectbox("Multiple clusters match this ID, choose one", hit_clusters, key="cluster_from_id")
    st.sidebar.write(f"{len(hits)} samples matched." if len(hits) else "No samples matched this ID search.")

cluster_from_hg = None
if meta is not None:
    search_haplo = st.sidebar.text_input("Search haplogroup (mt or Y)")
    if search_haplo:
        cols_hg = [c for c in df_samples.columns if "haplogroup" in c.lower()]
        if cols_hg:
            mask = pd.Series(False, index=df_samples.index)
            for c in cols_hg:
                mask = mask | df_samples[c].astype(str).str.contains(search_haplo.strip(), case=False, na=False, regex=False)
            hits_hg = df_samples[mask]
            hit_clusters_hg = sorted(hits_hg["cluster"].unique())
            if len(hit_clusters_hg) == 1:
                cluster_from_hg = hit_clusters_hg[0]
            elif len(hit_clusters_hg) > 1:
                cluster_from_hg = st.sidebar.selectbox("Multiple clusters match this haplogroup, choose one", hit_clusters_hg, key="cluster_from_hg")
            st.sidebar.write(f"{len(hits_hg)} samples matched." if len(hits_hg) else "No samples matched this haplogroup search.")

default_cluster = "All"
if cluster_from_id:
    default_cluster = cluster_from_id
elif cluster_from_hg:
    default_cluster = cluster_from_hg

selected = st.sidebar.selectbox("Select cluster", ["All"] + clusters, index=(["All"] + clusters).index(default_cluster))

col_fav_add, col_fav_clear = st.sidebar.columns(2)
with col_fav_add:
    if st.button("★ Add to favorites") and selected != "All":
        st.session_state["favorites"].add(selected)
with col_fav_clear:
    if st.button("Clear favorites"):
        st.session_state["favorites"] = set()
if st.session_state["favorites"]:
    st.sidebar.markdown("**Favorite clusters:**")
    for c in sorted(st.session_state["favorites"]):
        st.sidebar.write(f"- {c}")

# ───────────────────────── Main columns ─────────────────────────

left, right = st.columns([1.1, 1.2])

with left:
    if selected != "All":
        samples_view = df_samples[df_samples["cluster"] == selected].copy()
    else:
        samples_view = df_samples.copy()
    st.subheader("Samples in current view")
    st.dataframe(samples_view, use_container_width=True, height=300)

    st.subheader("Pedigree notes console")
    sample_choices = samples_view["sample"].tolist()
    selected_sample_note = st.selectbox(
        "Select sample to append to notes",
        sample_choices if sample_choices else [""],
        key="sample_to_add_notes",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Add sample to notes") and sample_choices:
            row = samples_view[samples_view["sample"] == selected_sample_note].iloc[0]
            line = make_note_line(row)
            if line not in st.session_state["pedigree_notes"]:
                st.session_state["pedigree_notes"] += line + "\n"
    with c2:
        if st.button("Add all cluster samples") and not samples_view.empty:
            lines = [make_note_line(r) for _, r in samples_view.iterrows()]
            existing = set(filter(None, st.session_state["pedigree_notes"].splitlines()))
            for line in lines:
                if line not in existing:
                    st.session_state["pedigree_notes"] += line + "\n"
                    existing.add(line)
    with c3:
        if st.button("Clear notes"):
            st.session_state["pedigree_notes"] = ""
    st.text_area("Notes", key="pedigree_notes", height=200)
    st.download_button("Download notes.txt", st.session_state["pedigree_notes"].encode("utf-8"),
                       file_name="pedigree_notes.txt", mime="text/plain")

with right:
    st.subheader("Pairwise relationships")

    cm_min_val = float(df["total_cM"].min())
    cm_max_val = float(df["total_cM"].max())
    cm_default = min(1000.0, cm_max_val)

    # Slider + manual number input in sync
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        min_cm_slider = st.slider(
            "Min total IBD (cM)",
            min_value=cm_min_val,
            max_value=cm_max_val,
            value=cm_default,
            step=10.0,
            key="min_cm_slider",
        )
    with filter_col2:
        min_cm_input = st.number_input(
            "Or type exact cM",
            min_value=cm_min_val,
            max_value=cm_max_val,
            value=min_cm_slider,
            step=1.0,
            key="min_cm_input",
        )

    # Use manual input if it differs from slider
    min_cm = min_cm_input if min_cm_input != min_cm_slider else min_cm_slider

    only_cluster = df.copy()
    if selected != "All":
        nodes_sel = [n for n, c in cluster_map.items() if c == selected]
        only_cluster = only_cluster[
            only_cluster["sample1"].isin(nodes_sel) & only_cluster["sample2"].isin(nodes_sel)
        ]
    only_cluster = only_cluster[only_cluster["total_cM"] >= min_cm].copy()

    st.dataframe(
        only_cluster[["sample1", "sample2", "total_cM", "relationship_class"]].sort_values("total_cM", ascending=False),
        use_container_width=True, height=250,
    )

    if selected != "All":
        nodes = [s for s in G.nodes() if cluster_map[s] == selected]
        H = G.subgraph(nodes)
        title = f"IBD network ({selected})"
    else:
        H = G
        title = "IBD network (All clusters)"

    if H.number_of_nodes() == 0:
        st.warning("No nodes to display for this selection.")
    else:
        try:
            pos = nx.spring_layout(H, weight="weight", seed=42)
        except Exception:
            pos = nx.circular_layout(H)

        node_x, node_y, text_labels, hovertext = [], [], [], []
        for n, (px, py) in pos.items():
            node_x.append(px)
            node_y.append(py)
            text_labels.append(n if selected != "All" else "")
            label = n
            if meta is not None and n in meta.index:
                if "haplogroup_mt" in meta.columns:
                    label += f" | mt: {meta.loc[n, 'haplogroup_mt']}"
                if "haplogroup_y" in meta.columns:
                    label += f" | Y: {meta.loc[n, 'haplogroup_y']}"
            hovertext.append(label)

        edge_x, edge_y = [], []
        for u, v in H.edges():
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                  text=text_labels if selected != "All" else None,
                                  textposition="top center",
                                  marker=dict(size=10, color="steelblue"),
                                  hovertext=hovertext, hoverinfo="text"))
        fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False),
                           showlegend=False, height=680, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

if segments_df is not None:
    st.subheader("Unified segments preview")
    st.dataframe(segments_df.head(200), use_container_width=True, height=200)
