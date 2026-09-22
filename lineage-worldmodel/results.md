# Lineage-supervised differentiation world model — stopped

## Data gate

**STOPPED_DATA_GATE.** No model or proxy experiment was run.

The target is the Weinreb et al. (2020) LARRY hematopoiesis dataset, GEO **GSE134242**. The public companion `paper-data` documentation confirms that the in-vitro experiment has days 2, 4, and 6, a cell-by-gene expression matrix, cell metadata, and a cell-by-clone membership matrix. The original LARRY repository documents the clone-assignment workflow. However, the directly downloadable normalized expression matrix is approximately 1.97 GB compressed, and the prepared AnnData download is approximately 2.78 GB. The GEO supplementary file list exposes expression count tables but does not expose the clone-assignment matrix.

I also checked the MEGATRON public LARRY subset, which documents a smaller 3,221-cell / 365-clone dataset with the needed clone-resolved inputs. Its input files are hosted behind a Mega folder; the folder was discoverable, but the files were not directly obtainable within this run.

## Local verification

The only local data artifact retained is the GSE134242 supplementary file list. It confirms many public expression TSVs, but it does not provide a locally processable expression-plus-clone pair. Therefore clone count, cells-per-clone distribution, timepoint overlap, gene count, and barcode dropout could not be computed honestly from a joined dataset.

## What was not run

- No ancestor/descendant clone pairs were fabricated.
- No MLP, baselines, held-out-clone split, metrics, or Go/No-Go result was run.
- No proxy was substituted for clone-resolved lineage supervision.

The data gate is therefore unresolved within reasonable effort, so the requested experiment stops here.
