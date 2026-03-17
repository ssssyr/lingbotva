# LingBot-VA MT50 Setup Notes

This repository does not ship a ready-made MetaWorld data adapter. After reading
the current training code, the practical path for MT50 fine-tuning is:

1. Create an isolated LingBot environment with the LingBot, LeRobot, and
   MetaWorld dependencies.
2. Convert your MetaWorld demonstrations into the LeRobot format expected by
   `wan_va/dataset/lerobot_latent_dataset.py`.
3. Pre-extract video latents and create `empty_emb.pt`.
4. Fine-tune with the new `mt50_train` config.

## Convert Raw MT50 To LeRobot

After collecting and cleaning raw MT50 episodes under `metaworld/data/mt50_raw`,
run:

```bash
source /home/syr/anaconda3/etc/profile.d/conda.sh
conda activate lingbot-mt50

python metaworld/convert_mt50_to_lerobot.py \
  --input-root metaworld/data/mt50_raw \
  --output-root metaworld/data/mt50_lerobot \
  --video-mode hardlink
```

This conversion script:

- renumbers episodes into one contiguous LeRobot dataset
- writes `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`,
  and `meta/episodes_stats.jsonl`
- creates one parquet file per episode under `data/chunk-*/`
- materializes source videos under `videos/chunk-*/`

Notes:

- `--video-mode hardlink` avoids duplicating the mp4 payload when source and
  destination are on the same filesystem. If hardlinking fails, the script
  falls back to copying.
- The script only prepares LeRobot metadata, parquet rows, and videos. It does
  not create `latents/` or `empty_emb.pt`.

## Extract Latents And Empty Embedding

After `mt50_lerobot` is ready, extract Wan latents and create `empty_emb.pt`:

```bash
source /home/syr/anaconda3/etc/profile.d/conda.sh
conda activate lingbot-mt50

export LINGBOT_VA_MODEL_PATH=/path/to/lingbot-va-base

python metaworld/extract_mt50_lerobot_latents.py \
  --dataset-root metaworld/data/mt50_lerobot \
  --model-path "${LINGBOT_VA_MODEL_PATH}" \
  --device cuda:0
```

This script will:

- read `meta/episodes.jsonl`
- encode each episode segment for every selected camera into `latents/`
- write `empty_emb.pt` at the dataset root

Defaults:

- it uses all video cameras listed in `meta/info.json`
- it keeps the LeRobot fps unchanged, so MT50 stays aligned with
  `LINGBOT_VA_ACTION_PER_FRAME=1`
- it skips latent files that already exist unless `--overwrite` is given

## What The Current Code Expects

The fine-tuning pipeline in `wan_va/train.py` and
`wan_va/dataset/lerobot_latent_dataset.py` assumes:

- The dataset root contains one or more LeRobot datasets with `meta/info.json`.
- `meta/episodes.jsonl` includes an `action_config` field for each episode.
- Video latents already exist under `latents/`, mirroring the LeRobot `videos/`
  layout.
- The model always consumes a 30-D action tensor. MT50's 4-D action should be
  mapped into LingBot's canonical 30-D layout.

The new `wan_va/configs/va_mt50_cfg.py` template maps the MT50 action to:

- `0, 1, 2`: Cartesian delta action
- `28`: gripper action

Unused channels are padded with zero by the existing dataset code.

## Create The Environment

Run:

```bash
bash script/create_mt50_env.sh lingbot-mt50
conda activate lingbot-mt50
python script/check_mt50_env.py
```

Optional:

```bash
INSTALL_FLASH_ATTN=1 bash script/create_mt50_env.sh lingbot-mt50
```

Notes:

- `script/create_mt50_env.sh` installs `lerobot==0.3.3` with `--no-deps`.
  This matches the upstream LingBot README and avoids the current
  `lerobot` metadata conflict with `torch==2.9.0`.
- Training mode requires `attn_mode="flex"` in
  `<model>/transformer/config.json`.
- Inference mode requires `attn_mode="torch"` or `attn_mode="flashattn"`.
- On headless Linux, keep `MUJOCO_GL=egl`.

## Expected Dataset Layout

Point `LINGBOT_VA_DATASET_PATH` at a directory shaped like:

```text
/path/to/mt50_lerobot_dataset/
  empty_emb.pt
  meta/
    info.json
    episodes.jsonl
  videos/
  latents/
```

The camera key defaults to `observation.images.main`. Override it with:

```bash
export LINGBOT_VA_MT50_CAMERA_KEY=observation.images.corner2
```

If your video frame rate is lower than the action rate, set
`LINGBOT_VA_ACTION_PER_FRAME` to the number of action steps aligned to each
latent frame.

## Run A Small Fine-Tuning Test

Example:

```bash
export LINGBOT_VA_TRAIN_MODEL_PATH=/path/to/lingbot-va-base
export LINGBOT_VA_DATASET_PATH=/path/to/mt50_lerobot_dataset
export LINGBOT_VA_ENABLE_WANDB=0

NGPU=1 \
LINGBOT_VA_BATCH_SIZE=1 \
LINGBOT_VA_GRAD_ACCUM_STEPS=4 \
LINGBOT_VA_NUM_STEPS=200 \
bash script/run_va_posttrain_mt50.sh
```

## Remaining Work Outside This Repository

This patch only prepares the LingBot side. You still need to supply:

- MetaWorld MT50 rollouts or demonstrations.
- A conversion step from MetaWorld episodes to LeRobot.
- Latent extraction compatible with the Wan VAE.
- Dataset-specific action statistics if `[-1, 1]` normalization is too loose for
  your data distribution.
