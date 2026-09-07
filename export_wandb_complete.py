"""Consolidate, download, standardize, and verify all W&B logs and loss metrics.

Ensures that every trained model weight in the export folder has:
1. Complete Weights & Biases logs (config, summary, execution logs).
2. Complete metric histories with both train and val loss metrics.
3. Master loss & weights mapping inventory in CSV and JSON formats.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

WORKSPACE_DIR = Path(__file__).parent.resolve()
EXPORT_DIR = WORKSPACE_DIR / "export"
EXPORT_WANDB_DIR = EXPORT_DIR / "wandb"
EXPORT_RUNS_DIR = EXPORT_WANDB_DIR / "runs"
EXPORT_HISTORIES_DIR = EXPORT_WANDB_DIR / "metric_histories"
EXTERNAL_EXPORT_DIR = Path(r"C:\Users\emil_brezovsky\Documents\GitHub\export")

EXPORT_WANDB_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_HISTORIES_DIR.mkdir(parents=True, exist_ok=True)

# Master Model Registry Mapping (40 Trained Best Weights + 4 Pretrain Backbones)
MODELS_REGISTRY: list[dict[str, Any]] = [
    # --- Ultralytics Baselines (3 Seeds) ---
    {
        "weight_file": "weights/rtdetr-l_seed_42_best.pt",
        "model_family": "RT-DETR",
        "variant": "rtdetr-l",
        "seed": 42,
        "run_id": "zu66apkg",
        "run_name": "rtdetr-l_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260720_193352-zu66apkg",
        "existing_history": "rtdetr-l_seed_42_zu66apkg_history.csv",
    },
    {
        "weight_file": "weights/rtdetr-l_seed_100_best.pt",
        "model_family": "RT-DETR",
        "variant": "rtdetr-l",
        "seed": 100,
        "run_id": "3m3c4nev",
        "run_name": "rtdetr-l_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260723_091206-3m3c4nev",
        "existing_history": "rtdetr-l_seed_100_3m3c4nev_history.csv",
    },
    {
        "weight_file": "weights/rtdetr-l_seed_999_best.pt",
        "model_family": "RT-DETR",
        "variant": "rtdetr-l",
        "seed": 999,
        "run_id": "vved41t4",
        "run_name": "rtdetr-l_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260726_114223-vved41t4",
        "existing_history": "rtdetr-l_seed_999_vved41t4_history.csv",
    },
    {
        "weight_file": "weights/yolo11n_seed_42_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11n",
        "seed": 42,
        "run_id": "ar6i6zf9",
        "run_name": "yolo11n_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260704_014215-ar6i6zf9",
        "existing_history": "yolo11n_seed_42_ar6i6zf9_history.csv",
    },
    {
        "weight_file": "weights/yolo11n_seed_42_culled_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11n-culled",
        "seed": 42,
        "run_id": "5y4tf9cy",
        "run_name": "yolo11n_seed_42_culledset",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260703_005219-5y4tf9cy",
        "existing_history": "yolo11n_seed_42_culledset_5y4tf9cy_history.csv",
    },
    {
        "weight_file": "weights/yolo11n_seed_100_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11n",
        "seed": 100,
        "run_id": "anoo65tn",
        "run_name": "yolo11n_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260704_103502-anoo65tn",
        "existing_history": "yolo11n_seed_100_anoo65tn_history.csv",
    },
    {
        "weight_file": "weights/yolo11n_seed_999_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11n",
        "seed": 999,
        "run_id": "fm0l8sac",
        "run_name": "yolo11n_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260704_205714-fm0l8sac",
        "existing_history": "yolo11n_seed_999_fm0l8sac_history.csv",
    },
    {
        "weight_file": "weights/yolo11s_seed_42_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11s",
        "seed": 42,
        "run_id": "4j0nt488",
        "run_name": "yolo11s_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260705_085819-4j0nt488",
        "existing_history": "yolo11s_seed_42_4j0nt488_history.csv",
    },
    {
        "weight_file": "weights/yolo11s_seed_100_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11s",
        "seed": 100,
        "run_id": "4eec0su5",
        "run_name": "yolo11s_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260705_184854-4eec0su5",
        "existing_history": "yolo11s_seed_100_4eec0su5_history.csv",
    },
    {
        "weight_file": "weights/yolo11s_seed_999_best.pt",
        "model_family": "YOLO11",
        "variant": "yolo11s",
        "seed": 999,
        "run_id": "z4rsmysx",
        "run_name": "yolo11s_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260706_044444-z4rsmysx",
        "existing_history": "yolo11s_seed_999_z4rsmysx_history.csv",
    },
    {
        "weight_file": "weights/yolo12n_seed_42_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12n",
        "seed": 42,
        "run_id": "ytafff8k",
        "run_name": "yolo12n_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260709_120305-ytafff8k",
        "existing_history": "yolo12n_seed_42_ytafff8k_history.csv",
    },
    {
        "weight_file": "weights/yolo12n_seed_100_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12n",
        "seed": 100,
        "run_id": "jvobnsss",
        "run_name": "yolo12n_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260710_004729-jvobnsss",
        "existing_history": "yolo12n_seed_100_jvobnsss_history.csv",
    },
    {
        "weight_file": "weights/yolo12n_seed_999_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12n",
        "seed": 999,
        "run_id": "v1hqqnxx",
        "run_name": "yolo12n_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260710_132235-v1hqqnxx",
        "existing_history": "yolo12n_seed_999_v1hqqnxx_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_seed_42_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s",
        "seed": 42,
        "run_id": "9d6dvly1",
        "run_name": "yolo12s_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260711_070054-9d6dvly1",
        "existing_history": "yolo12s_seed_42_9d6dvly1_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_seed_100_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s",
        "seed": 100,
        "run_id": "enum9pts",
        "run_name": "yolo12s_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260712_015652-enum9pts",
        "existing_history": "yolo12s_seed_100_enum9pts_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_seed_999_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s",
        "seed": 999,
        "run_id": "wgxnkj0p",
        "run_name": "yolo12s_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260712_194125-wgxnkj0p",
        "existing_history": "yolo12s_seed_999_wgxnkj0p_history.csv",
    },
    {
        "weight_file": "weights/yolo26n_seed_42_best.pt",
        "model_family": "YOLO26",
        "variant": "yolo26n",
        "seed": 42,
        "run_id": "jbb75ehr",
        "run_name": "yolo26n_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260706_134055-jbb75ehr",
        "existing_history": "yolo26n_seed_42_jbb75ehr_history.csv",
    },
    {
        "weight_file": "weights/yolo26n_seed_100_best.pt",
        "model_family": "YOLO26",
        "variant": "yolo26n",
        "seed": 100,
        "run_id": "m8qd8vxg",
        "run_name": "yolo26n_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260706_230641-m8qd8vxg",
        "existing_history": "yolo26n_seed_100_m8qd8vxg_history.csv",
    },
    {
        "weight_file": "weights/yolo26n_seed_999_best.pt",
        "model_family": "YOLO26",
        "variant": "yolo26n",
        "seed": 999,
        "run_id": "mh02wc72",
        "run_name": "yolo26n_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260707_093217-mh02wc72",
        "existing_history": "yolo26n_seed_999_mh02wc72_history.csv",
    },
    {
        "weight_file": "weights/yolo26s_seed_42_best.pt",
        "model_family": "YOLO26",
        "variant": "yolo26s",
        "seed": 42,
        "run_id": "kqgtkgh0",
        "run_name": "yolo26s_seed_42",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260708_022702-kqgtkgh0",
        "existing_history": "yolo26s_seed_42_kqgtkgh0_history.csv",
    },
    {
        "weight_file": "weights/yolo26s_seed_100_best.pt",
        "model_family": "YOLO26",
        "variant": "yolo26s",
        "seed": 100,
        "run_id": "m1jdeeqb",
        "run_name": "yolo26s_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260708_131621-m1jdeeqb",
        "existing_history": "yolo26s_seed_100_m1jdeeqb_history.csv",
    },
    {
        "weight_file": "weights/yolo26s_seed_999_best.pt",
        "model_family": "YOLO26",
        "variant": "yolo26s",
        "seed": 999,
        "run_id": "0oizbnve",
        "run_name": "yolo26s_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260709_012018-0oizbnve",
        "existing_history": "yolo26s_seed_999_0oizbnve_history.csv",
    },
    # --- Fine-Tuned Distillation: YOLO12s + LVD-1689M (3 Seeds) ---
    {
        "weight_file": "weights/yolo12s_lvd1689m_seed42_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-distill-lvd1689m",
        "seed": 42,
        "run_id": "17la41ve",
        "run_name": "finetune_distill_lvd1689m_to_yolo12s",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260817_103609-17la41ve",
        "existing_history": "finetune_distill_lvd1689m_to_yolo12s_17la41ve_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_lvd1689m_seed100_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-distill-lvd1689m",
        "seed": 100,
        "run_id": "bt6acvgo",
        "run_name": "finetune_distill_lvd1689m_to_yolo12s_seed_100",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260826_214630-bt6acvgo",
        "existing_history": "finetune_distill_lvd1689m_to_yolo12s_seed_100_bt6acvgo_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_lvd1689m_seed999_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-distill-lvd1689m",
        "seed": 999,
        "run_id": "zq4z9wuf",
        "run_name": "finetune_distill_lvd1689m_to_yolo12s_seed_999",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260827_023949-zq4z9wuf",
        "existing_history": "finetune_distill_lvd1689m_to_yolo12s_seed_999_zq4z9wuf_history.csv",
    },
    # --- Fine-Tuned Distillation: YOLO12s + SAT-493M (3 Seeds) ---
    {
        "weight_file": "weights/yolo12s_sat493m_seed42_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-distill-sat493m",
        "seed": 42,
        "run_id": "boiftxcq",
        "run_name": "finetune_distill_sat493m_to_yolo12s",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260815_144627-boiftxcq",
        "existing_history": "finetune_distill_sat493m_to_yolo12s_boiftxcq_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_sat493m_seed100_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-distill-sat493m",
        "seed": 100,
        "run_id": "cwmwbgcw",
        "run_name": "finetune_distill_sat493m_to_yolo12s_seed_100",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260827_091309-cwmwbgcw",
        "existing_history": "finetune_distill_sat493m_to_yolo12s_seed_100_cwmwbgcw_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_sat493m_seed999_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-distill-sat493m",
        "seed": 999,
        "run_id": "z2hsaoor",
        "run_name": "finetune_distill_sat493m_to_yolo12s_seed_999",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260827_150743-z2hsaoor",
        "existing_history": "finetune_distill_sat493m_to_yolo12s_seed_999_z2hsaoor_history.csv",
    },
    # --- Fine-Tuned Control: YOLO12s Stock Undistilled (3 Seeds) ---
    {
        "weight_file": "weights/yolo12s_stock_seed42_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-stock",
        "seed": 42,
        "run_id": "runs_detect_train_s42",
        "run_name": "yolo12s_stock_seed_42",
        "project": "_baseline",
        "local_runs_dir": "runs/detect/train",
        "existing_results_csv": "runs/detect/train/results.csv",
    },
    {
        "weight_file": "weights/yolo12s_stock_seed100_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-stock",
        "seed": 100,
        "run_id": "enum9pts",
        "run_name": "yolo12s_seed_100",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260712_015652-enum9pts",
        "existing_history": "yolo12s_seed_100_enum9pts_history.csv",
    },
    {
        "weight_file": "weights/yolo12s_stock_seed999_best.pt",
        "model_family": "YOLO12",
        "variant": "yolo12s-stock",
        "seed": 999,
        "run_id": "wgxnkj0p",
        "run_name": "yolo12s_seed_999",
        "project": "_baseline",
        "local_wandb_src": "GitHub_export:run-20260712_194125-wgxnkj0p",
        "existing_history": "yolo12s_seed_999_wgxnkj0p_history.csv",
    },
    # --- Fine-Tuned Distillation: ViT-T + LVD-1689M (3 Seeds) ---
    {
        "weight_file": "weights/vitt16_lvd1689m_seed42_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-distill-lvd1689m",
        "seed": 42,
        "run_id": "tmokmwo8",
        "run_name": "finetune_distill_lvd1689m_to_dinov3_vitt16",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260821_004901-tmokmwo8",
        "existing_history": "finetune_distill_lvd1689m_to_dinov3_vitt16_tmokmwo8_history.csv",
    },
    {
        "weight_file": "weights/vitt16_lvd1689m_seed100_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-distill-lvd1689m",
        "seed": 100,
        "run_id": "bgluhvqs",
        "run_name": "finetune_distill_lvd1689m_to_dinov3_vitt16_seed_100",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260825_110307-bgluhvqs",
        "existing_history": "finetune_distill_lvd1689m_to_dinov3_vitt16_seed_100_bgluhvqs_history.csv",
    },
    {
        "weight_file": "weights/vitt16_lvd1689m_seed999_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-distill-lvd1689m",
        "seed": 999,
        "run_id": "rp9l391j",
        "run_name": "finetune_distill_lvd1689m_to_dinov3_vitt16_seed_999",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260825_182453-rp9l391j",
        "existing_history": "finetune_distill_lvd1689m_to_dinov3_vitt16_seed_999_rp9l391j_history.csv",
    },
    # --- Fine-Tuned Distillation: ViT-T + SAT-493M (3 Seeds) ---
    {
        "weight_file": "weights/vitt16_sat493m_seed42_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-distill-sat493m",
        "seed": 42,
        "run_id": "distill_sat493m_vitt16_s42",
        "run_name": "finetune_distill_sat493m_to_dinov3_vitt16",
        "project": "_baseline",
        "tfevents_path": "runs/distill/distill_sat493m_to_dinov3_vitt16/finetune/events.out.tfevents.1787100219.CF-LILA-004-D.72568.1",
        "train_log_path": "runs/distill/distill_sat493m_to_dinov3_vitt16/finetune/train.log",
    },
    {
        "weight_file": "weights/vitt16_sat493m_seed100_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-distill-sat493m",
        "seed": 100,
        "run_id": "9ao7sutk",
        "run_name": "dinov3_vitt16_ltdetr_seed_100_rtdetrv2_9ao7sutk",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260821_082853-9ao7sutk",
        "existing_history": "dinov3_vitt16_ltdetr_seed_100_rtdetrv2_9ao7sutk_9ao7sutk_history.csv",
    },
    {
        "weight_file": "weights/vitt16_sat493m_seed999_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-distill-sat493m",
        "seed": 999,
        "run_id": "5koevthq",
        "run_name": "dinov3_vitt16_ltdetr_seed_999_rtdetrv2_5koevthq",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260821_215056-5koevthq",
        "existing_history": "dinov3_vitt16_ltdetr_seed_999_rtdetrv2_5koevthq_5koevthq_history.csv",
    },
    # --- Fine-Tuned Control: ViT-T Stock Undistilled (3 Seeds) ---
    {
        "weight_file": "weights/vitt16_stock_seed42_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-stock",
        "seed": 42,
        "run_id": "wdjvply0",
        "run_name": "dinov3_vitt16_seed_42_dfine",
        "project": "_baseline",
        "downloaded_src": "wandb_downloaded_logs/11gourf8_dinov3_vitt16_seed_42_dfine",
        "existing_history": "dinov3_vitt16_seed_42_dfine_wdjvply0_history.csv",
    },
    {
        "weight_file": "weights/vitt16_stock_seed100_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-stock",
        "seed": 100,
        "run_id": "vtd49szt",
        "run_name": "dinov3_vitt16_seed_100_dfine",
        "project": "_baseline",
        "downloaded_src": "wandb_downloaded_logs/vtd49szt_dinov3_vitt16_seed_100_dfine",
        "existing_history": "dinov3_vitt16_seed_100_dfine_vtd49szt_history.csv",
    },
    {
        "weight_file": "weights/vitt16_stock_seed999_best.pt",
        "model_family": "DINOv3 ViT-T",
        "variant": "dinov3-vitt16-stock",
        "seed": 999,
        "run_id": "lxjiuf03",
        "run_name": "dinov3_vitt16_seed_999_dfine",
        "project": "_baseline",
        "downloaded_src": "wandb_downloaded_logs/lxjiuf03_dinov3_vitt16_seed_999_dfine",
        "existing_history": "dinov3_vitt16_seed_999_dfine_lxjiuf03_history.csv",
    },
    # --- Pretrained Distillation Backbones (Stage 1) ---
    {
        "weight_file": "pretrained_backbones/distill_lvd1689m_to_dinov3_vitt16_last.pt",
        "model_family": "DINOv3 ViT-T Backbone",
        "variant": "distill-lvd1689m-vitt16",
        "seed": 42,
        "run_id": "yaonckxh",
        "run_name": "pretrain_distill_lvd1689m_to_dinov3_vitt16",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260819_205451-yaonckxh",
        "existing_history": "pretrain_distill_lvd1689m_to_dinov3_vitt16_yaonckxh_history.csv",
    },
    {
        "weight_file": "pretrained_backbones/distill_lvd1689m_to_yolo12s_last.pt",
        "model_family": "YOLO12s Backbone",
        "variant": "distill-lvd1689m-yolo12s",
        "seed": 42,
        "run_id": "uqjbn2us",
        "run_name": "pretrain_distill_lvd1689m_to_yolo12s",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260816_031343-uqjbn2us",
        "existing_history": "pretrain_distill_lvd1689m_to_yolo12s_uqjbn2us_history.csv",
    },
    {
        "weight_file": "pretrained_backbones/distill_sat493m_to_dinov3_vitt16_last.pt",
        "model_family": "DINOv3 ViT-T Backbone",
        "variant": "distill-sat493m-vitt16",
        "seed": 42,
        "run_id": "59yo5t4r",
        "run_name": "pretrain_distill_sat493m_to_dinov3_vitt16",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260817_164517-59yo5t4r",
        "existing_history": "pretrain_distill_sat493m_to_dinov3_vitt16_59yo5t4r_history.csv",
    },
    {
        "weight_file": "pretrained_backbones/distill_sat493m_to_yolo12s_last.pt",
        "model_family": "YOLO12s Backbone",
        "variant": "distill-sat493m-yolo12s",
        "seed": 42,
        "run_id": "x9pqsvzx",
        "run_name": "pretrain_distill_sat493m_to_yolo12s",
        "project": "_baseline",
        "local_wandb_src": "wandb:offline-run-20260811_152505-x9pqsvzx",
        "existing_history": "pretrain_distill_sat493m_to_yolo12s_x9pqsvzx_history.csv",
    },
]


def extract_tfevents_df(tfevent_path: Path) -> pd.DataFrame:
    """Extract scalar time-series from a TensorBoard tfevents file."""
    ea = EventAccumulator(str(tfevent_path))
    ea.Reload()
    tags = ea.Tags()["scalars"]
    data: dict[int, dict[str, Any]] = {}
    for tag in tags:
        for e in ea.Scalars(tag):
            step = int(e.step)
            if step not in data:
                data[step] = {"_step": step}
            data[step][tag] = float(e.value)
    df = pd.DataFrame(list(data.values())).sort_values("_step").reset_index(drop=True)
    return df


def extract_results_csv_df(results_csv_path: Path) -> pd.DataFrame:
    """Convert Ultralytics results.csv to standard metric history dataframe."""
    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" in df.columns:
        df["_step"] = df["epoch"]
    # Total losses
    train_components = [
        c
        for c in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
        if c in df.columns
    ]
    if train_components:
        df["train_loss"] = df[train_components].sum(axis=1)
    val_components = [
        c for c in ["val/box_loss", "val/cls_loss", "val/dfl_loss"] if c in df.columns
    ]
    if val_components:
        df["val_loss"] = df[val_components].sum(axis=1)
    return df


def consolidate_run_files(
    reg_entry: dict[str, Any],
    run_dir: Path,
    history_df: pd.DataFrame,
    api: wandb.Api,
) -> None:
    """Consolidates config, summary, and metadata files into target run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Always save the history.csv inside run_dir
    history_df.to_csv(run_dir / "history.csv", index=False)

    # 1. Check local_wandb_src
    src_spec = reg_entry.get("local_wandb_src")
    if src_spec:
        prefix, folder_name = src_spec.split(":", 1)
        if prefix == "GitHub_export":
            src_dir = EXTERNAL_EXPORT_DIR / "wandb" / folder_name
        else:
            src_dir = WORKSPACE_DIR / "wandb" / folder_name

        if src_dir.exists():
            for item in src_dir.iterdir():
                dest = run_dir / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

    # 2. Check downloaded_src
    dl_src = reg_entry.get("downloaded_src")
    if dl_src:
        src_dir = WORKSPACE_DIR / dl_src
        if src_dir.exists():
            for item in src_dir.iterdir():
                dest = run_dir / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

    # 3. Check local_runs_dir
    runs_src = reg_entry.get("local_runs_dir")
    if runs_src:
        src_dir = WORKSPACE_DIR / runs_src
        if src_dir.exists():
            args_f = src_dir / "args.yaml"
            if args_f.exists() and not (run_dir / "args.yaml").exists():
                shutil.copy2(args_f, run_dir / "args.yaml")

    # 4. Check if summary.json and config.json exist; if not, try to fetch via wandb API
    config_file = run_dir / "config.json"
    summary_file = run_dir / "summary.json"
    run_id = reg_entry.get("run_id")
    project = reg_entry.get("project", "_baseline")

    if (
        (not config_file.exists() or not summary_file.exists())
        and run_id
        and not run_id.startswith("runs_")
        and not run_id.startswith("distill_")
    ):
        try:
            r = api.run(f"brezo-boku-vienna/{project}/{run_id}")
            if not summary_file.exists():
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(r.summary._json_dict, f, indent=2)
            if not config_file.exists():
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(r.config, f, indent=2)
        except Exception as e:
            print(f"Note: Could not query W&B API for {run_id}: {e}")

    # If still no summary/config, generate from history
    if not summary_file.exists():
        synth_summary: dict[str, Any] = {
            "run_id": run_id,
            "run_name": reg_entry.get("run_name"),
            "model": reg_entry.get("variant"),
            "seed": reg_entry.get("seed"),
        }
        if "train_loss" in history_df.columns:
            t_s = [
                float(x)
                for x in history_df["train_loss"].dropna().to_list()
                if float(x) > 0
            ]
            synth_summary["train_loss_final"] = t_s[-1] if t_s else None
            synth_summary["train_loss_min"] = min(t_s) if t_s else None
        if "val_loss" in history_df.columns:
            v_s = [
                float(x)
                for x in history_df["val_loss"].dropna().to_list()
                if float(x) > 0
            ]
            synth_summary["val_loss_final"] = v_s[-1] if v_s else None
            synth_summary["val_loss_min"] = min(v_s) if v_s else None
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(synth_summary, f, indent=2)

    if not config_file.exists():
        synth_config = {
            "model": reg_entry.get("variant"),
            "seed": reg_entry.get("seed"),
            "project": project,
            "weight_file": reg_entry.get("weight_file"),
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(synth_config, f, indent=2)


def process_all_models() -> pd.DataFrame:
    """Processes all 44 models, standardizing metrics and populating export/wandb."""
    print("Connecting to Weights & Biases API...")
    api = wandb.Api()

    inventory_records: list[dict[str, Any]] = []

    for entry in MODELS_REGISTRY:
        weight_rel = entry["weight_file"]
        weight_stem = Path(weight_rel).stem.replace("_best", "")
        run_id = entry["run_id"]
        run_name = entry["run_name"]
        print(f"\nProcessing {weight_rel} ({run_id} - {run_name})...")

        # 1. Obtain history dataframe
        history_df: pd.DataFrame | None = None

        # Check existing_history in EXPORT_HISTORIES_DIR
        ex_hist = entry.get("existing_history")
        if ex_hist and (EXPORT_HISTORIES_DIR / ex_hist).exists():
            history_df = pd.read_csv(EXPORT_HISTORIES_DIR / ex_hist)
        elif (
            entry.get("tfevents_path")
            and (WORKSPACE_DIR / entry["tfevents_path"]).exists()
        ):
            history_df = extract_tfevents_df(WORKSPACE_DIR / entry["tfevents_path"])
        elif (
            entry.get("existing_results_csv")
            and (WORKSPACE_DIR / entry["existing_results_csv"]).exists()
        ):
            history_df = extract_results_csv_df(
                WORKSPACE_DIR / entry["existing_results_csv"]
            )
        else:
            # Try W&B online API
            try:
                r = api.run(
                    f"brezo-boku-vienna/{entry.get('project', '_baseline')}/{run_id}"
                )
                history_df = r.history()
            except Exception as e:
                print(f"Failed to fetch history from W&B API for {run_id}: {e}")

        if history_df is None or history_df.empty:
            raise RuntimeError(f"Could not load history for {weight_rel} ({run_id})")

        # 1. Compute total train loss from components if available
        train_loss_col = None
        val_loss_col = None

        u_components = [
            c
            for c in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
            if c in history_df.columns
        ]
        r_components = [
            c
            for c in ["train/l1_loss", "train/giou_loss", "train/cls_loss"]
            if c in history_df.columns
        ]
        lt_components = [c for c in history_df.columns if c.startswith("train_loss/")]

        if u_components:
            history_df["train_loss"] = history_df[u_components].sum(axis=1, min_count=1)
            train_loss_col = "train_loss"
        elif r_components:
            history_df["train_loss"] = history_df[r_components].sum(axis=1, min_count=1)
            train_loss_col = "train_loss"
        elif lt_components:
            history_df["train_loss"] = history_df[lt_components].sum(
                axis=1, min_count=1
            )
            train_loss_col = "train_loss"
        elif "train_loss" in history_df.columns:
            train_loss_col = "train_loss"

        # 2. Compute total val loss from components if available
        u_val_components = [
            c
            for c in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
            if c in history_df.columns
        ]
        r_val_components = [
            c
            for c in ["val/l1_loss", "val/giou_loss", "val/cls_loss"]
            if c in history_df.columns
        ]
        lt_val_components = [c for c in history_df.columns if c.startswith("val_loss/")]

        if u_val_components:
            history_df["val_loss"] = history_df[u_val_components].sum(
                axis=1, min_count=1
            )
            val_loss_col = "val_loss"
        elif r_val_components:
            history_df["val_loss"] = history_df[r_val_components].sum(
                axis=1, min_count=1
            )
            val_loss_col = "val_loss"
        elif lt_val_components:
            history_df["val_loss"] = history_df[lt_val_components].sum(
                axis=1, min_count=1
            )
            val_loss_col = "val_loss"
        elif "val_loss" in history_df.columns:
            val_loss_col = "val_loss"

        # Check performance metrics
        map50: float | None = None
        for col in ["metrics/mAP50(B)", "val_metric/map_50", "mAP50"]:
            if col in history_df.columns:
                m_list = [float(x) for x in history_df[col].dropna().to_list()]
                if m_list:
                    map50 = max(m_list)
                    break

        map50_95: float | None = None
        for col in ["metrics/mAP50-95(B)", "val_metric/map", "mAP50_95"]:
            if col in history_df.columns:
                m_list = [float(x) for x in history_df[col].dropna().to_list()]
                if m_list:
                    map50_95 = max(m_list)
                    break

        # Calculate train/val loss summary numbers
        train_vals: list[float] = []
        if train_loss_col and train_loss_col in history_df.columns:
            train_vals = [
                float(x)
                for x in history_df[train_loss_col].dropna().to_list()
                if float(x) > 0
            ]

        val_vals: list[float] = []
        if val_loss_col and val_loss_col in history_df.columns:
            val_vals = [
                float(x)
                for x in history_df[val_loss_col].dropna().to_list()
                if float(x) > 0
            ]

        train_loss_final = train_vals[-1] if train_vals else None
        train_loss_min = min(train_vals) if train_vals else None
        val_loss_final = val_vals[-1] if val_vals else None
        val_loss_min = min(val_vals) if val_vals else None

        # Save standard history CSV to metric_histories
        # 1. Standard weight-stem named CSV
        stem_history_csv = EXPORT_HISTORIES_DIR / f"{weight_stem}_history.csv"
        history_df.to_csv(stem_history_csv, index=False)

        # 2. Original run history CSV if present
        if ex_hist:
            history_df.to_csv(EXPORT_HISTORIES_DIR / ex_hist, index=False)

        # Create run folder in export/wandb/runs/
        safe_name = "".join(
            [c if c.isalnum() or c in ("-", "_") else "_" for c in run_name]
        )
        run_folder = EXPORT_RUNS_DIR / f"{run_id}_{safe_name}"
        consolidate_run_files(entry, run_folder, history_df, api)

        rel_history_path = f"wandb/metric_histories/{stem_history_csv.name}"
        rel_run_path = f"wandb/runs/{run_folder.name}"

        steps_count = len(history_df)
        t_min_str = f"{train_loss_min:.4f}" if train_loss_min is not None else "N/A"
        t_fin_str = f"{train_loss_final:.4f}" if train_loss_final is not None else "N/A"
        v_min_str = f"{val_loss_min:.4f}" if val_loss_min is not None else "N/A"
        v_fin_str = f"{val_loss_final:.4f}" if val_loss_final is not None else "N/A"
        m50_str = f"{map50:.4f}" if map50 is not None else "N/A"

        print(f" -> Train Loss: min={t_min_str}, final={t_fin_str}")
        print(f" -> Val Loss:   min={v_min_str}, final={v_fin_str}")
        print(f" -> Steps/Epochs: {steps_count}, mAP50: {m50_str}")

        inventory_records.append(
            {
                "model_family": entry["model_family"],
                "variant": entry["variant"],
                "seed": entry["seed"],
                "weight_file": entry["weight_file"],
                "wandb_run_id": run_id,
                "wandb_run_name": run_name,
                "wandb_project": entry.get("project", "_baseline"),
                "train_loss_min": round(train_loss_min, 5)
                if train_loss_min is not None
                else None,
                "train_loss_final": round(train_loss_final, 5)
                if train_loss_final is not None
                else None,
                "val_loss_min": round(val_loss_min, 5)
                if val_loss_min is not None
                else None,
                "val_loss_final": round(val_loss_final, 5)
                if val_loss_final is not None
                else None,
                "best_mAP50": round(map50, 5) if map50 is not None else None,
                "best_mAP50_95": round(map50_95, 5) if map50_95 is not None else None,
                "total_recorded_steps": steps_count,
                "history_csv": rel_history_path,
                "run_directory": rel_run_path,
            }
        )

    inv_df = pd.DataFrame(inventory_records)

    # Save inventory to CSV & JSON
    inv_csv = EXPORT_WANDB_DIR / "loss_and_weights_inventory.csv"
    inv_json = EXPORT_WANDB_DIR / "loss_and_weights_inventory.json"
    inv_df.to_csv(inv_csv, index=False)
    with open(inv_json, "w", encoding="utf-8") as f:
        json.dump(inventory_records, f, indent=2)

    print(f"\nSaved inventory of {len(inv_df)} models to:")
    print(f" - {inv_csv}")
    print(f" - {inv_json}")

    return inv_df


def generate_wandb_readme(inv_df: pd.DataFrame) -> None:
    """Generates comprehensive README.md inside export/wandb."""
    readme_path = EXPORT_WANDB_DIR / "README.md"
    lines = [
        "# 📊 Weights & Biases Experiment Logs & Loss Metrics Inventory",
        "",
        "This directory contains complete offline Weights & Biases logs, metric histories, configurations,",
        "and training/validation loss curves for every trained model checkpoint and backbone in the export package.",
        "",
        "---",
        "",
        "## 1. Directory Overview",
        "",
        "```text",
        "export/wandb/",
        "├── loss_and_weights_inventory.csv    # Master CSV index mapping every weight to its train/val loss & W&B run",
        "├── loss_and_weights_inventory.json   # Machine-readable JSON inventory with loss statistics and paths",
        "├── runs/                             # Complete per-run W&B execution directories (config, summary, history, logs)",
        "│   ├── <run_id>_<run_name>/",
        "│   │   ├── config.json / config.yaml # Hyperparameters, optimizer settings, dataset paths",
        "│   │   ├── summary.json              # Final summary metrics",
        "│   │   ├── history.csv               # Epoch/step loss time-series",
        "│   │   └── output.log / .wandb       # Console outputs and binary run logs",
        "│   └── ...",
        "├── metric_histories/                 # Standalone full CSV time-series for all models and runs",
        "│   ├── <weight_stem>_history.csv     # Direct 1-to-1 match for every weight checkpoint",
        "│   └── ...",
        "├── runs_summary.csv                  # Complete tabular summary across all benchmark runs",
        "└── runs_summary.json                 # Comprehensive JSON summary across all runs",
        "```",
        "",
        "---",
        "",
        "## 2. Master Loss & Performance Metrics by Model Checkpoint",
        "",
        "| Model Family | Variant | Seed | Weight Checkpoint | W&B Run ID | Best Train Loss | Final Train Loss | Best Val Loss | Final Val Loss | Best mAP50 | Recorded Steps |",
        "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    records: list[dict[str, Any]] = inv_df.to_dict(orient="records")  # pyright: ignore[reportAssignmentType]
    for row in records:
        t_min_v = row.get("train_loss_min")
        t_min = (
            f"{float(t_min_v):.4f}"
            if t_min_v is not None and not pd.isna(t_min_v)
            else "N/A"
        )

        t_fin_v = row.get("train_loss_final")
        t_fin = (
            f"{float(t_fin_v):.4f}"
            if t_fin_v is not None and not pd.isna(t_fin_v)
            else "N/A"
        )

        v_min_v = row.get("val_loss_min")
        v_min = (
            f"{float(v_min_v):.4f}"
            if v_min_v is not None and not pd.isna(v_min_v)
            else "N/A"
        )

        v_fin_v = row.get("val_loss_final")
        v_fin = (
            f"{float(v_fin_v):.4f}"
            if v_fin_v is not None and not pd.isna(v_fin_v)
            else "N/A"
        )

        m50_v = row.get("best_mAP50")
        m50 = (
            f"{float(m50_v) * 100:.2f}%"
            if m50_v is not None and not pd.isna(m50_v)
            else "N/A"
        )

        weight_name = Path(str(row["weight_file"])).name

        lines.append(
            f"| {row['model_family']} | `{row['variant']}` | {row['seed']} | [`{weight_name}`](../{row['weight_file']}) | `{row['wandb_run_id']}` | {t_min} | {t_fin} | {v_min} | {v_fin} | {m50} | {row['total_recorded_steps']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Metric History File Mapping",
            "",
            "Every weight in `weights/` has a direct corresponding metric history file located in `metric_histories/<weight_stem>_history.csv`.",
            "To inspect loss progression in Python:",
            "",
            "```python",
            "import pandas as pd",
            "",
            "# Load history for fine-tuned ViT-T (SAT-493M pretraining, Seed 42)",
            "df = pd.read_csv('export/wandb/metric_histories/vitt16_sat493m_seed42_history.csv')",
            "print(df[['_step', 'train_loss', 'val_loss', 'val_metric/map_50']].dropna().tail())",
            "```",
            "",
            "For YOLO models:",
            "```python",
            "# Load history for YOLO12s (Seed 100)",
            "df = pd.read_csv('export/wandb/metric_histories/yolo12s_seed_100_history.csv')",
            "print(df[['_step', 'train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss', 'metrics/mAP50(B)']].dropna().tail())",
            "```",
        ]
    )

    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated documentation: {readme_path}")


def sync_to_external_export() -> None:
    """Syncs the completed wandb artifacts to C:\\Users\\emil_brezovsky\\Documents\\GitHub\\export\\wandb."""
    if not EXTERNAL_EXPORT_DIR.exists():
        print(
            f"External export directory not found at {EXTERNAL_EXPORT_DIR}, skipping sync."
        )
        return

    ext_wandb = EXTERNAL_EXPORT_DIR / "wandb"
    ext_wandb.mkdir(parents=True, exist_ok=True)
    print(f"\nSynchronizing complete W&B logs and loss metrics to: {ext_wandb}...")

    # 1. Sync inventory files
    for fn in [
        "loss_and_weights_inventory.csv",
        "loss_and_weights_inventory.json",
        "README.md",
    ]:
        src = EXPORT_WANDB_DIR / fn
        if src.exists():
            shutil.copy2(src, ext_wandb / fn)

    # 2. Sync metric_histories
    ext_hist = ext_wandb / "metric_histories"
    ext_hist.mkdir(parents=True, exist_ok=True)
    for f in EXPORT_HISTORIES_DIR.glob("*.csv"):
        shutil.copy2(f, ext_hist / f.name)

    # 3. Sync runs
    ext_runs = ext_wandb / "runs"
    ext_runs.mkdir(parents=True, exist_ok=True)
    for rdir in EXPORT_RUNS_DIR.iterdir():
        if rdir.is_dir():
            target_rdir = ext_runs / rdir.name
            if not target_rdir.exists():
                shutil.copytree(rdir, target_rdir)
            else:
                for f in rdir.iterdir():
                    dest_f = target_rdir / f.name
                    if not dest_f.exists():
                        if f.is_dir():
                            shutil.copytree(f, dest_f)
                        else:
                            shutil.copy2(f, dest_f)

    # 4. Sync missing weights to external export if not present
    ext_weights = EXTERNAL_EXPORT_DIR / "weights"
    ext_weights_best = ext_weights / "best"
    ext_weights_best.mkdir(parents=True, exist_ok=True)

    src_weights = EXPORT_DIR / "weights"
    for pt in src_weights.glob("*.pt"):
        dest_pt = ext_weights_best / pt.name
        if not dest_pt.exists():
            print(
                f"Copying missing weight {pt.name} to external export/weights/best/..."
            )
            shutil.copy2(pt, dest_pt)

    # Also sync pretrained backbones
    src_pb = EXPORT_DIR / "pretrained_backbones"
    ext_pb = EXTERNAL_EXPORT_DIR / "pretrained_backbones"
    ext_pb.mkdir(parents=True, exist_ok=True)
    if src_pb.exists():
        for pt in src_pb.glob("*.pt"):
            dest_pt = ext_pb / pt.name
            if not dest_pt.exists():
                print(
                    f"Copying backbone {pt.name} to external export/pretrained_backbones/..."
                )
                shutil.copy2(pt, dest_pt)

    print("Synchronization to external export directory complete!")


def main() -> None:
    print("=================================================================")
    print("       W&B LOGS & LOSS METRICS COMPLETE EXPORTER & SYNC          ")
    print("=================================================================")
    inv_df = process_all_models()
    generate_wandb_readme(inv_df)
    sync_to_external_export()
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
