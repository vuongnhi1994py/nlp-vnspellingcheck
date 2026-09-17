# Vietnamese Spelling Check

Final model: **PP-OCRv5 Mobile Recognition fine-tuned for 4 epochs** on `vi_rec_100k`.

## Dataset and setup

- Dataset: `vi_rec_100k` (`rec_train.txt`, `rec_val.txt`, `rec_test.txt`, `vi_dict.txt`)
- Reported test set: `rec_test.txt`, 3003 samples for every model
- Base model: PP-OCRv5 Mobile Recognition
- Python 3.12.13; PaddlePaddle GPU 3.2.0; NVIDIA T4 x2
- PaddleOCR official repository, branch `main`, commit `2661c7c0ef5c613e8f93c6e93b2e052399f0f854`
- Seed: 1024; final training time: 18:55:50
- Final checkpoint: `https://drive.google.com/file/d/10Eq6WjvMlQzu3b5MMwGtDVCLF8bVGX2f/view?usp=sharing`

The exact training configs from the backup are in `configs/vi_PP-OCRv5_mobile_rec_epoch2.yml` and `configs/vi_PP-OCRv5_mobile_rec_epoch4.yml`. Both use `vi_dict.txt` and seed 1024. The 4-epoch run resumes the 2-epoch checkpoint. The config values differ only in `epoch_num` and the output directory; the model and other training settings match. The original fine-tune template remains in `configs/vi_PP-OCRv5_mobile_rec.yml`.

## Test results

| Model | Samples | Accuracy | Normalized edit distance | Training time |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR original | 3003 | 0.042291 | 0.795104 | 0 |
| Fine-tuned PP-OCRv5 Mobile (4 epochs) | 3003 | 0.363303 | 0.951337 | 18:55:50 |

## 2 vs 4 epoch experiment

| Run | Samples | Accuracy | Normalized edit distance |
| --- | ---: | ---: | ---: |
| 2 epochs | 3003 | 0.311022 | 0.947753 |
| 4 epochs | 3003 | 0.363303 | 0.951337 |

With the same training settings and test set, accuracy rose from 0.311022 to 0.363303 and normalized edit distance rose from 0.947753 to 0.951337. The larger accuracy gain suggests many nearly correct predictions became exact matches. Both metrics improved, so the 4-epoch run is the final model.

## Error analysis

Among 20 classified errors from the 4-epoch model: tone/diacritic errors 11/20 (55%), character errors 9/20 (45%), repeated-character errors 0/20 (0%). See `results/error_analysis_20_classified.csv`.

## Artifacts and evaluation

`results/pred_test_baseline.jsonl`, `pred_test_epoch2.jsonl`, `pred_test_epoch4.jsonl`, and `pred_test_finetuned.jsonl` each contain 3003 records with `image`, `gt`, and `pred` fields. `pred_test_finetuned.jsonl` is identical to the 4-epoch file. Summary tables are `results/model_comparison.csv` and `results/epoch2_vs_epoch4.csv`.

Rebuild the small artifacts from the read-only backup and evaluate them with Python's standard library:

```sh
python src/finalize_artifacts.py
python eval.py results/pred_test_baseline.jsonl
python eval.py results/pred_test_finetuned.jsonl
```

The final checkpoint link is pending. Checkpoints, pretrained weights, datasets, and large inference models are kept outside this repository.
