# genealogic-IBD-clusters

Experimental pipeline to explore genealogical relationships and putative IBD segments between modern samples and ancient DNA samples from the Akbari/AADR panel, using genomic similarity results and archaeological metadata.

## Goals

- Detect IBD or high-similarity clusters between modern IBS individuals and ancient West Eurasian samples. [file:7]
- Annotate each match with basic archaeological information (culture, chronology, region) when available.
- Prioritize combinations with sufficient genomic coverage to avoid false positives in very low-coverage ancient samples. [file:5]
- Produce outputs that are easy to reuse for genealogical analysis and visualizations (relationship graphs, heatmaps, etc.). [file:7]

## Input data

- Similarity / IBD file: `IBS_biosamples_IBD_akbari.txt`, with rows of the form  
  `HGxxxx Iyyyyy.panel score`, where `score` is a genomic similarity or IBD measure between the modern biosample and the ancient sample. [file:7]
- Coriell IBS catalog: `Coriell-Catalog-Export-03-22-2026_IBS.csv`, to obtain sex, family role, and basic pedigree information for IBS individuals. [file:6]
- Akbari coverage table: `Akbari_coverage.txt`, with number of missing genotypes and fraction missing (`F_MISS`) for each ancient sample. [file:5]

## Workflow

1. ID normalization  
   - Clean and harmonize IDs for modern samples (HGxxxx) and ancient samples (Ixxxx.AG, Ixxxx.TW, etc.). [file:7]  
   - Filter or flag ancient samples with very high `F_MISS`, using a configurable threshold. [file:5]

2. Similarity matrix construction  
   - Reshape `IBS_biosamples_IBD_akbari.txt` into a matrix where rows = modern IBS samples and columns = ancient samples, with `score` as the cell value. [file:7]  
   - Optionally export this matrix as CSV/TSV for downstream analyses.

3. Cluster detection  
   - For each IBS, rank ancient samples by `score` and retain the top N candidates. [file:7]  
   - Apply quality filters (minimum coverage, panel type, etc.). [file:5]

4. Genealogical annotation  
   - Link each IBS to its Coriell record (family structure, role in the pedigree) to contextualize IBD clusters. [file:6]  
   - Prepare output tables with: IBS ID, ancient ID, score, F_MISS, and relevant metadata fields where available. [file:7][file:5]

## Expected outputs

- CSV tables with:
  - IBD clusters per IBS (one row per IBS–ancient pair). [file:7]
  - Per-IBS summaries listing the best ancient candidates and their quality metrics. [file:7][file:5]
- Auxiliary metadata files (harmonized ID lists, applied filters) for reuse in other scripts.

## Current status

- This version focuses on building core tables and ID harmonization; it does not yet integrate the full AADR archaeological metadata or automatic visualizations. [file:174]
- The pipeline is tuned to the Akbari 1240K panel with ~15.8k ancient samples and ~1.2M SNPs. [file:5]

## Usage notes

- Always manually review matches with very high scores involving ancient samples that have low coverage or inconsistent metadata. [file:5]
- The repository is intended as an exploratory tool for advanced genealogy; results should not be used as clinical or legal evidence.
