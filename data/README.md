# Synthetic Condominium Resident Request Datasets

All messages in this folder are **100% synthetic** and authored specifically for this Jev System-One triage project. No real resident names, personal identifiers, or private estate records appear in any file.

## Files

| File | Rows | Purpose |
| --- | --- | --- |
| `train.csv` | 160 | Balanced training split (16 examples per category across all 10 categories) used to fit the Platt-calibrated `LocalJevSimulatorBackend`. |
| `evaluation.csv` | 60 | Disjoint evaluation benchmark covering 5 slices: `standard` (45), `ambiguous` (5), `emergency` (4), `multi_intent` (3), and `adversarial` (3). |
| `batch_sample.csv` | 6 | Ready-to-upload sample CSV for testing the Batch Triage tab and `--batch-csv` CLI flag. |

## Integrity & Leakage Prevention

`condo_jev.evaluation.verify_no_train_eval_leakage()` verifies that zero normalized messages overlap between `train.csv` and `evaluation.csv` and records SHA-256 dataset digests in `reports/jev-evaluation-report.json`.
