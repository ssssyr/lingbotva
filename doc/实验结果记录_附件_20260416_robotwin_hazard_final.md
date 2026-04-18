# 2026-04-16 / 2026-04-17 RobotWin Hazard Scheduler Final 附件

本附件给出该次实验的完整任务级结果表，供论文写表格、补充材料或回查原始数据时使用。

原始结果目录：

- 结果根目录：[robotwin_hazard_final_eval5_20260416](/home/syr/code/lingbot-va/RoboTwin/results/robotwin_hazard_final_eval5_20260416)
- 任务级指标目录：[metrics](/home/syr/code/lingbot-va/RoboTwin/results/robotwin_hazard_final_eval5_20260416/stseed-10000/metrics)

说明：

- `Success` 为该任务 5 次测试中的成功次数
- `Steps Mean` 为当次评测记录的 `scheduler_video_steps_mean`
- `Video Runtime Mean (ms)` 为当次评测记录的 `scheduler_video_runtime_ms_mean`
- 该批评测中的步数/时延统计来自 2026-04-16 当晚结果；2026-04-17 已修复 online `hazard_jump_v2` 的记账语义，因此论文终稿前建议在修复后重新跑一版完整评测确认最终数字

| Task | Success | Rate | Steps Mean | Video Runtime Mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| adjust_bottle | 5/5 | 100.0% | 7.30 | 629.3 |
| beat_block_hammer | 5/5 | 100.0% | 6.30 | 534.4 |
| blocks_ranking_rgb | 5/5 | 100.0% | 7.40 | 668.2 |
| blocks_ranking_size | 4/5 | 80.0% | 7.46 | 708.4 |
| click_alarmclock | 5/5 | 100.0% | 7.53 | 629.1 |
| click_bell | 5/5 | 100.0% | 6.53 | 545.6 |
| dump_bin_bigbin | 5/5 | 100.0% | 7.32 | 618.3 |
| grab_roller | 5/5 | 100.0% | 7.35 | 617.8 |
| handover_block | 5/5 | 100.0% | 11.26 | 954.7 |
| handover_mic | 4/5 | 80.0% | 6.67 | 603.3 |
| hanging_mug | 2/5 | 40.0% | 7.08 | 689.5 |
| lift_pot | 5/5 | 100.0% | 9.75 | 808.6 |
| move_can_pot | 4/5 | 80.0% | 8.97 | 767.4 |
| move_pillbottle_pad | 5/5 | 100.0% | 7.54 | 648.0 |
| move_playingcard_away | 5/5 | 100.0% | 7.77 | 662.9 |
| move_stapler_pad | 2/5 | 40.0% | 7.22 | 647.0 |
| open_laptop | 5/5 | 100.0% | 8.54 | 729.6 |
| open_microwave | 4/5 | 80.0% | 9.25 | 905.8 |
| pick_diverse_bottles | 5/5 | 100.0% | 7.60 | 648.0 |
| pick_dual_bottles | 5/5 | 100.0% | 6.50 | 560.2 |
| place_a2b_left | 5/5 | 100.0% | 8.68 | 739.3 |
| place_a2b_right | 4/5 | 80.0% | 8.14 | 708.7 |
| place_bread_basket | 5/5 | 100.0% | 9.10 | 790.4 |
| place_bread_skillet | 5/5 | 100.0% | 6.28 | 551.1 |
| place_burger_fries | 5/5 | 100.0% | 7.95 | 689.6 |
| place_can_basket | 5/5 | 100.0% | 11.10 | 946.7 |
| place_cans_plasticbox | 5/5 | 100.0% | 5.05 | 467.4 |
| place_container_plate | 4/5 | 80.0% | 8.88 | 764.8 |
| place_dual_shoes | 5/5 | 100.0% | 9.72 | 840.2 |
| place_empty_cup | 5/5 | 100.0% | 8.07 | 685.2 |
| place_fan | 5/5 | 100.0% | 7.19 | 614.5 |
| place_mouse_pad | 5/5 | 100.0% | 7.15 | 630.5 |
| place_object_basket | 5/5 | 100.0% | 8.70 | 748.3 |
| place_object_scale | 5/5 | 100.0% | 7.65 | 654.7 |
| place_object_stand | 5/5 | 100.0% | 10.16 | 850.5 |
| place_phone_stand | 5/5 | 100.0% | 7.56 | 645.5 |
| place_shoe | 5/5 | 100.0% | 7.34 | 657.4 |
| press_stapler | 5/5 | 100.0% | 8.36 | 712.3 |
| put_bottles_dustbin | 4/5 | 80.0% | 8.41 | 827.9 |
| put_object_cabinet | 5/5 | 100.0% | 13.00 | 1086.1 |
| rotate_qrcode | 4/5 | 80.0% | 7.43 | 642.5 |
| scan_object | 4/5 | 80.0% | 6.51 | 577.1 |
| shake_bottle | 5/5 | 100.0% | 8.60 | 712.9 |
| shake_bottle_horizontally | 5/5 | 100.0% | 8.85 | 737.8 |
| stack_blocks_three | 5/5 | 100.0% | 7.22 | 657.7 |
| stack_blocks_two | 5/5 | 100.0% | 7.58 | 666.8 |
| stack_bowls_three | 4/5 | 80.0% | 7.08 | 675.1 |
| stack_bowls_two | 5/5 | 100.0% | 7.56 | 658.4 |
| stamp_seal | 5/5 | 100.0% | 8.73 | 736.6 |
| turn_switch | 2/5 | 40.0% | 10.09 | 872.3 |
