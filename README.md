# Case-specific error bounds for contralateral bone reconstruction

Code for *Case-Specific Error Bounds for Contralateral Bone Reconstruction: External Validation in Post-Mortem CT*.

The distal third of the left femur and tibia is removed and reconstructed from the mirrored right bone. Three 95% error bounds (fixed, residual-scaled and regression on near-defect fit with CV+ calibration) are derived in a development cohort and tested in an independent external cohort.

## Data

Both datasets are open access and are downloaded by `reproduce.py`.

- Development: OpenSkeleton (New Mexico Decedent Image Database), https://doi.org/10.5281/zenodo.23592083. The 102 decedents documented in the dataset metadata are analysed.
- External validation: Virtual Skeleton Database lower-limb bone models, https://github.com/MCM-Fischer/VSDFullBodyBoneModels. Four subjects with arthroplasty, intraosseous access or extensive metal artifact are excluded, leaving 26.

## Usage

```
pip install -r requirements.txt
python reproduce.py
```

Results are written to `results/`. Random seeds are fixed.

| File | Content |
| --- | --- |
| `results_main.csv` | One row per reconstruction: length, global and near-defect fit, HD95 and mean surface distance, standard and refined registration |
| `results_defect_size.csv` | Reconstructions at 25%, 33% and 50% defects |
| `table1_bounds.csv` | Internal and external coverage and width of each bound |
| `refinement.csv` | Paired comparison of standard and refined registration |
| `defect_size.csv` | HD95 by defect size |
| `summary.txt` | All reported values |
| `figure1`, `figure2` | Figures (PNG and PDF) |

## Files

- `mirroring.py`: reconstruction, error measurement, error bounds and figures
- `reproduce.py`: downloads the data and runs all analyses

## License

MIT. The data remain under their original licences.
