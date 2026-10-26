# 1. Data processing and features

**Responsible: 孙煜皓.** Download the official audio archive, verify metadata/checksums and extract reproducible features.

- `download_dataset.py`: resumable download, archive MD5 verification and safe extraction.
- `download_parallel.py`: optional parallel-download utility; the main documented route uses the standard downloader.
- `validate_metadata.py`: compare the official metadata with the Soundata index and record source-ID overlaps.
- `extract_features.py`: verify each clip and build an 8,732 x 40 MFCC feature matrix.
- `data/`: public metadata, checksum index and source attribution. Raw WAV files are downloaded locally.
- `results/`: the clip manifest, metadata audit and class distribution used by the experiment.

Processing: mono -> 22,050 Hz -> 20 MFCCs -> temporal mean and standard deviation -> 40 features. The next stage reads `features/features.npz`; feature caches are excluded from Git.

See the [root README](../README.md) for commands. Run this stage before model training on a new computer.
