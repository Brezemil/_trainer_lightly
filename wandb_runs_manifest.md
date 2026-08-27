# 📑 W&B Offline Runs Manifest & Master 3-Seed Performance Matrix

This document provides a comprehensive, re-audited manifest of all **Weights & Biases (W&B) offline execution runs** logged in this repository (wandb/) and evaluated on the independent hold-out test set (=510$ tiles).

---

## 1. Multi-Seed Performance Audit Matrix (Hold-Out Test Set)

| Architecture / Model Setup | Seed 42 | Seed 100 | Seed 999 | 3-Seed Mean mAP50 | mAP(50-95) Mean | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Distilled ViT-T** (lvd1689m) | **51.24%** 🏆 | 38.41% | 41.82% | **43.82% ± 6.64%** | 20.43% | ✅ 3/3 Seeds Done |
| **Satellite Distilled ViT-T** (sat493m) | 46.70% | 42.01% | 39.13% | **42.61% ± 3.82%** | 20.24% | ✅ 3/3 Seeds Done |
| **Stock Undistilled ViT-T Baseline** | 38.34% | 38.49% | 39.92% | **38.92% ± 0.87%** | 18.74% | ✅ 3/3 Seeds Done |
| **YOLO12s Stock Undistilled** | 34.38% | 47.51% | 46.97% | **42.95% ± 7.43%** | 20.13% | ✅ 3/3 Seeds Done |
| **YOLO12s Satellite Distilled** (sat493m) | 47.33% | Pending | Pending | 47.33% (Seed 42) | 22.18% | ⏳ Stage 1 Pretrained |
| **YOLO12s General Distilled** (lvd1689m) | 45.43% | Pending | Pending | 45.43% (Seed 42) | 20.59% | ⏳ Stage 1 Pretrained |

---

## 2. Executive Inventory Summary

| Category | Count | Directory Path | W&B Cloud Status |
| :--- | :---: | :--- | :---: |
| **Valid Completed Runs** | **129** | wandb/ | ✅ Synced to W&B Cloud |
| **Archived Aborted Runs** | **84** | wandb/aborted_archive/ | 🛡️ Safely Isolated |
| **TOTAL RUNS AUDITED** | **213** | — | — |

---

## 3. Evaluation Checkpoint File Mapping

| Model Setup | Seed | Test Metric File | mAP50 | mAP(50-95) |
| :--- | :---: | :--- | :---: | :---: |
| **ViT-T Undistilled** | 42 | dinov3_vitt16_seed_42_dfine_TEST_coco_metrics.json | 38.34% | 18.52% |
| **ViT-T Undistilled** | 100 | dinov3_vitt16_seed_100_dfine_TEST_coco_metrics.json | 38.49% | 18.34% |
| **ViT-T Undistilled** | 999 | dinov3_vitt16_seed_999_dfine_TEST_coco_metrics.json | 39.92% | 19.37% |
| **ViT-T Satellite Distilled** | 100 | dinov3_vitt16_seed_100_sat493m_TEST_coco_metrics.json | 42.01% | 19.38% |
| **ViT-T Satellite Distilled** | 999 | dinov3_vitt16_seed_999_sat493m_TEST_coco_metrics.json | 39.13% | 19.35% |
| **ViT-T General Distilled** | 100 | inetune_distill_lvd1689m_to_dinov3_vitt16_seed100_TEST_coco_metrics.json | 38.41% | 17.70% |
| **ViT-T General Distilled** | 999 | inetune_distill_lvd1689m_to_dinov3_vitt16_seed999_TEST_coco_metrics.json | 41.82% | 19.62% |
| **YOLO12s Stock** | 42 | stock_yolo12s_seed_42_TEST_coco_metrics.json | 34.38% | 14.58% |
| **YOLO12s Stock** | 100 | stock_yolo12s_seed_100_TEST_coco_metrics.json | 47.51% | 22.77% |
| **YOLO12s Stock** | 999 | stock_yolo12s_seed_999_TEST_coco_metrics.json | 46.97% | 23.05% |
