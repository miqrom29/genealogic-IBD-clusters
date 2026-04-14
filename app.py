import io
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="IBD Cluster Notes Demo", layout="wide")
st.title("IBD Cluster Notes Demo")
st.caption("Classic IBD cluster mode + optional multi-CSV genealogy loader + pedigree notes console")

# ───────────────────────── Shared helpers ─────────────────────────

def clean_id(x: str) -> str:
    s = str(x)
    if s.startswith("[") and "](" in s:
        return s.split("[", 1)[1].split("]", 1)[0]
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
    return f"{row['sample']} | mt={mt} | Y={y} | {row['cluster']}"


def detect_separator(sample_bytes: bytes) -> str:
    text = sample_bytes.decode("utf-8", errors="ignore")
    if text.count(";") > text.count(","):
        return ";"
    return ","


def norm_float(s: str) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (float, int)):
        return float(s)
    s = str(s).strip().replace("‎", "")
    s = s.replace(" ", "")
    s = s.strip('"').strip("'")
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
        m = re.search(r"\(([0-9]+(?:\.[0-9]+)?)\s*cM", s)
        if m:
            return float(m.group(1))
    return None

# ───────────────────────── Multi-loader parsers ─────────────────────────

def parse_529_segments(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.rename(columns={
        "Name": "id1",
        "Match name": "id2",
        "Chromosome": "chrom",
        "Start point": "start",
        "End point": "end",
        "Genetic distance": "cM",
        "# SNPs": "snps",
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
        "Chromosome": "chrom",
        "Start of segment": "start",
        "Length of segment": "end",
        "Number of SNPs": "snps",
        "Length in centimorgan (cM)": "cM",
        "Type of segment": "segment_type",
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
    df2 = df[[name_col, total_cm_col]].copy()
    df2["total_cM"] = df2[total_cm_col].apply(norm_float)
    df2["id1"] = focal
    df2.rename(columns={name_col: "id2"}, inplace=True)
    df2["platform"] = "MyHeritage"
    df2["source_file"] = source
    pairs = df2[["id1", "id2", "total_cM", "platform", "source_file"]].dropna(subset=["total_cM"])
    segs = pd.DataFrame(columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"])
    return pairs, segs


def detect_and_parse(file, focal_sample: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    raw_bytes = file.getvalue()
    sep = detect_separator(raw_bytes)
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, dtype=str)
    cols_lower = [c.lower() for c in df.columns]
    if "name" in cols_lower and "match name" in cols_lower and "genetic distance" in cols_lower:
        pairs, segs = parse_529_segments(df, file.name)
        platform = "529andYou (segments)"
    elif "public name" in cols_lower and "length in centimorgan (cm)" in cols_lower:
        pairs, segs = parse_geneanet_segments(df, file.name)
        platform = "Geneanet (segments)"
    elif any("total de cm compartits" in c.lower() for c in df.columns) or any("total of shared dna" in c.lower() for c in df.columns):
        pairs, segs = parse_myheritage_matches(df, file.name, focal_sample)
        platform = "MyHeritage (matches)"
    else:
        platform = "Unknown/unsupported (yet)"
        pairs = pd.DataFrame(columns=["id1", "id2", "total_cM", "platform", "source_file"])
        segs = pd.DataFrame(columns=["id1", "id2", "chrom", "start", "end", "cM", "snps", "platform", "source_file"])
    return pairs, segs, platform

# ───────────────────────── Sidebar upload modes ─────────────────────────

if "favorites" not in st.session_state:
    st.session_state["favorites"] = set()
if "pedigree_notes" not in st.session_state:
    st.session_state["pedigree_notes"] = ""

st.sidebar.header("Input mode")
input_mode = st.sidebar.radio(
    "Choose input source",
    [
        "Classic IBD pairs file",
        "Multi-CSV genealogy loader",
    ],
)

st.sidebar.header("Optional sample metadata")
meta_file = st.sidebar.file_uploader(
    "Metadata CSV (sample,haplogroup_mt,mt_haplogroup,y_haplogroup,...)",
    type=["csv"],
    key="meta",
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

# ───────────────────────── Build pair table depending on mode ─────────────────────────

df = None
segments_df = None

if input_mode == "Classic IBD pairs file":
    st.sidebar.header("Upload IBD pairs")
    ibd_file = st.sidebar.file_uploader(
        "CSV or TSV with id1,id2,total_cM", type=["csv", "tsv"], key="ibd"
    )
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
    focal_sample = st.sidebar.text_input("Focal sample for match-only files", value="Encarna Vicente")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV files from DNA platforms",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_csv",
    )
    if not uploaded_files:
        st.info("Upload one or more genealogy CSV files to build a unified pair table.")
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
        file_name="unified_pairs_for_ibd_app.csv",
        mime="text/csv",
    )
    if segments_df is not None:
        st.download_button(
            "Download unified segments CSV",
            segments_df.to_csv(index=False).encode("utf-8"),
            file_name="unified_segments_for_ibd_app.csv",
            mime="text/csv",
        )

df["relationship_class"] = df["total_cM"].apply(classify_relationship)

st.subheader(f"Pairwise IBD input ({source_label})")
st.dataframe(
    df[["sample1", "sample2", "total_cM", "relationship_class"]].sort_values("total_cM", ascending=False),
    use_container_width=True,
    height=260,
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
    row = {
        "sample": node,
        "cluster": cluster_map[node],
        "partners": len(partners),
        "max_pairwise_cM": max(totals) if totals else 0.0,
    }
    if meta is not None and node in meta.index:
        for col in meta.columns:
            row[col] = meta.loc[node, col]
    rows.append(row)

df_samples = pd.DataFrame(rows).sort_values(["cluster", "sample"])
st.subheader("Sample assignments")
st.dataframe(df_samples, use_container_width=True, height=260)

st.sidebar.header("Cluster view")
clusters = sorted(df_samples["cluster"].unique())
search_id = st.sidebar.text_input("Search sample ID (partial ok)")
cluster_from_id = None
if search_id:
    hits = df_samples[
        df_samples["sample"].astype(str).str.contains(search_id.strip(), case=False, na=False, regex=False)
    ]
    hit_clusters = sorted(hits["cluster"].unique())
    if len(hit_clusters) == 1:
        cluster_from_id = hit_clusters[0]
    elif len(hit_clusters) > 1:
        cluster_from_id = st.sidebar.selectbox("Multiple clusters match this ID, choose one", hit_clusters, key="cluster_from_id")
    st.sidebar.write(f"{len(hits)} samples matched this ID search." if len(hits) else "No samples matched this ID search.")

search_haplo = None
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
            st.sidebar.write(f"{len(hits_hg)} samples matched this haplogroup search." if len(hits_hg) else "No samples matched this haplogroup search.")

default_cluster = "All"
if cluster_from_id:
    default_cluster = cluster_from_id
elif cluster_from_hg:
    default_cluster = cluster_from_hg

selected = st.sidebar.selectbox("Select cluster", ["All"] + clusters, index=(["All"] + clusters).index(default_cluster))

left, right = st.columns([1.1, 1.2])
with left:
    if selected != "All":
        samples_view = df_samples[df_samples["cluster"] == selected].copy()
    else:
        samples_view = df_samples.copy()
    st.subheader("Samples in current view")
    st.dataframe(samples_view, use_container_width=True, height=320)

    st.subheader("Pedigree notes console")
    sample_choices = samples_view["sample"].tolist()
    selected_sample_note = st.selectbox("Select sample to append to notes", sample_choices if sample_choices else [""], key="sample_to_add_notes")
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
    st.text_area("Notes", key="pedigree_notes", height=220)
    st.download_button("Download notes.txt", st.session_state["pedigree_notes"].encode("utf-8"), file_name="pedigree_notes.txt", mime="text/plain")

with right:
    st.subheader("Pairwise relationships")
    min_default = 1000.0 if float(df["total_cM"].max()) >= 1000 else float(df["total_cM"].min())
    min_cm = st.slider("Minimum total IBD (cM) to display", min_value=float(df["total_cM"].min()), max_value=float(df["total_cM"].max()), value=float(min_default), step=50.0)
    only_cluster = df.copy()
    if selected != "All":
        nodes_sel = [n for n, c in cluster_map.items() if c == selected]
        only_cluster = only_cluster[only_cluster["sample1"].isin(nodes_sel) & only_cluster["sample2"].isin(nodes_sel)]
    only_cluster = only_cluster[only_cluster["total_cM"] >= min_cm].copy()
    st.dataframe(only_cluster[["sample1", "sample2", "total_cM", "relationship_class"]].sort_values("total_cM", ascending=False), use_container_width=True, height=250)

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
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text_labels if selected != "All" else None, textposition="top center", marker=dict(size=10, color="steelblue"), hovertext=hovertext, hoverinfo="text"))
        fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False, height=700, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

if segments_df is not None:
    st.subheader("Unified segments preview")
    st.dataframe(segments_df.head(200), use_container_width=True, height=220)
