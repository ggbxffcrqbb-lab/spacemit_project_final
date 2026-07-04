# Phase 5：缺陷数据源筛选记录

适用日期：`2026-06-26`

## 使用原则

| 原则 | 说明 |
|---|---|
| 自有现场数据优先 | 最终上线效果取决于 Muse Pi Pro 实际场景，而不是公开基准 |
| 公开数据只做 warm-start 或补充 | 防止域偏移过大导致模型学到错误纹理 |
| 先做检测再做分割/分级 | 首版应优先缩短标注周期并打通板端部署 |

## 候选数据源

| 数据源 | 规模/特点 | 适合用途 | 不适合直接做什么 |
|---|---|---|---|
| Coating Defect Detection Dataset | 2697 张 RGB，风电塔筒涂装部件，已给 YOLO 边界框 | 作为首版涂层缺陷检测 warm-start | 不能直接等价为本项目三类缺陷语义 |
| Corrosion Condition State Semantic Segmentation Dataset | 440 张，腐蚀状态分割，四级状态 | 做腐蚀严重度辅助任务或分割参考 | 不适合作为首版 bbox 检测主数据 |
| Water-Based Coated Wood Products Dataset | 13400 张，划痕/裂纹/气泡/孔洞 | 学习“表面缺陷”通用纹理和小目标能力 | 木器漆与工业防腐表面域差较大 |
| MVTec AD | 5000+ 张，多类别工业异常定位 | 做 normal/abnormal 基线或异常检测预训练 | 不能直接输出本项目缺陷类别 |

## 当前决策

| 决策 | 说明 |
|---|---|
| 主训练集 | 以自有板端/现场图像为主 |
| warm-start 数据 | 优先试 `Coating Defect Detection Dataset` |
| 辅助任务 | 可把 `Corrosion Condition State` 数据用于后续严重度评估支线 |
| 非首版方向 | `MVTec AD` 和木器漆数据暂不直接混入主检测训练集 |
