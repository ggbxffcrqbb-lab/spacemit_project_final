from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.vision.image_utils import resize_long_edge
from app.vision.types import DefectCandidate


@dataclass(frozen=True)
class DirectDefectRule:
    project_label: str
    summary: str


DIRECT_DEFECT_RULES: dict[str, DirectDefectRule] = {
    "rust": DirectDefectRule(
        "rust_like_corrosion",
        "模型标签直接命中锈蚀/腐蚀语义，优先复核金属表面锈蚀和氧化情况。",
    ),
    "rust_spot": DirectDefectRule(
        "rust_like_corrosion",
        "模型标签直接命中锈蚀/腐蚀语义，优先复核金属表面锈蚀和氧化情况。",
    ),
    "corrosion": DirectDefectRule(
        "rust_like_corrosion",
        "模型标签直接命中锈蚀/腐蚀语义，优先复核金属表面锈蚀和氧化情况。",
    ),
    "corroded": DirectDefectRule(
        "rust_like_corrosion",
        "模型标签直接命中锈蚀/腐蚀语义，优先复核金属表面锈蚀和氧化情况。",
    ),
    "oxidation": DirectDefectRule(
        "rust_like_corrosion",
        "模型标签直接命中锈蚀/腐蚀语义，优先复核金属表面锈蚀和氧化情况。",
    ),
    "flaking": DirectDefectRule(
        "coating_flaking_or_delamination",
        "模型标签直接命中漆层剥落/分层语义，优先复核附着力失效和基材暴露情况。",
    ),
    "peeling": DirectDefectRule(
        "coating_flaking_or_delamination",
        "模型标签直接命中漆层剥落/分层语义，优先复核附着力失效和基材暴露情况。",
    ),
    "delamination": DirectDefectRule(
        "coating_flaking_or_delamination",
        "模型标签直接命中漆层剥落/分层语义，优先复核附着力失效和基材暴露情况。",
    ),
    "paint_flaking": DirectDefectRule(
        "coating_flaking_or_delamination",
        "模型标签直接命中漆层剥落/分层语义，优先复核附着力失效和基材暴露情况。",
    ),
    "chalking": DirectDefectRule(
        "chalking_or_powdering",
        "模型标签直接命中粉化/失光语义，优先复核表层老化和粉末化情况。",
    ),
    "powdering": DirectDefectRule(
        "chalking_or_powdering",
        "模型标签直接命中粉化/失光语义，优先复核表层老化和粉末化情况。",
    ),
    "cui": DirectDefectRule(
        "cui_risk_visual_hint",
        "模型标签直接命中 CUI 相关语义，优先复核保温层下腐蚀风险区域。",
    ),
    "cui_risk": DirectDefectRule(
        "cui_risk_visual_hint",
        "模型标签直接命中 CUI 相关语义，优先复核保温层下腐蚀风险区域。",
    ),
    "under_insulation_corrosion": DirectDefectRule(
        "cui_risk_visual_hint",
        "模型标签直接命中 CUI 相关语义，优先复核保温层下腐蚀风险区域。",
    ),
    "rust_like_corrosion": DirectDefectRule(
        "rust_like_corrosion",
        "模型标签已直接使用项目锈蚀语义，无需二次映射。",
    ),
    "coating_flaking_or_delamination": DirectDefectRule(
        "coating_flaking_or_delamination",
        "模型标签已直接使用项目漆层剥落语义，无需二次映射。",
    ),
    "chalking_or_powdering": DirectDefectRule(
        "chalking_or_powdering",
        "模型标签已直接使用项目粉化语义，无需二次映射。",
    ),
    "cui_risk_visual_hint": DirectDefectRule(
        "cui_risk_visual_hint",
        "模型标签已直接使用项目 CUI 风险语义，无需二次映射。",
    ),
}


def build_heuristic_defect_candidates(
    rgb: np.ndarray,
    max_candidates: int,
) -> tuple[list[DefectCandidate], dict[str, Any]]:
    sample = resize_long_edge(rgb, 960)

    sample_f = sample.astype(np.float32) / 255.0
    r = sample_f[..., 0]
    g = sample_f[..., 1]
    b = sample_f[..., 2]
    mean = sample_f.mean(axis=2)
    channel_max = sample_f.max(axis=2)
    channel_min = sample_f.min(axis=2)
    saturation = channel_max - channel_min

    grad_x = np.abs(np.diff(mean, axis=1, append=mean[:, -1:]))
    grad_y = np.abs(np.diff(mean, axis=0, append=mean[-1:, :]))
    edge_density = float(np.mean((grad_x + grad_y) > 0.18))
    texture_score = float(np.mean(grad_x + grad_y))

    rust_mask = (r > 0.32) & (r > g * 1.12) & (r > b * 1.28) & (saturation > 0.12)
    chalk_mask = (mean > 0.72) & (saturation < 0.12)
    bare_metal_mask = (mean > 0.6) & (saturation < 0.08)
    dark_spot_mask = (mean < 0.2) & (saturation < 0.18)

    rust_ratio = float(np.mean(rust_mask))
    chalk_ratio = float(np.mean(chalk_mask))
    bare_metal_ratio = float(np.mean(bare_metal_mask))
    dark_spot_ratio = float(np.mean(dark_spot_mask))
    low_saturation_ratio = float(np.mean(saturation < 0.1))

    rust_score = min(0.99, rust_ratio * 4.5 + dark_spot_ratio * 1.2 + edge_density * 0.8)
    flaking_score = min(
        0.99,
        bare_metal_ratio * 3.6 + edge_density * 1.5 + texture_score * 1.2,
    )
    chalking_score = min(
        0.99,
        chalk_ratio * 4.0 + low_saturation_ratio * 0.8 + max(0.0, 0.12 - texture_score) * 2.0,
    )
    cui_score = min(
        0.99,
        rust_ratio * 2.0 + dark_spot_ratio * 1.7 + edge_density * 0.7,
    )

    candidates = [
        DefectCandidate(
            label="rust_like_corrosion",
            score=rust_score,
            summary="画面中存在明显锈色区域，优先复核锈蚀、基材暴露或潮湿残留。",
            evidence={
                "rust_ratio": round(rust_ratio, 4),
                "dark_spot_ratio": round(dark_spot_ratio, 4),
                "edge_density": round(edge_density, 4),
            },
        ),
        DefectCandidate(
            label="coating_flaking_or_delamination",
            score=flaking_score,
            summary="亮色裸露区与边缘纹理较多，优先复核剥落、起皮或附着力失效。",
            evidence={
                "bare_metal_ratio": round(bare_metal_ratio, 4),
                "edge_density": round(edge_density, 4),
                "texture_score": round(texture_score, 4),
            },
        ),
        DefectCandidate(
            label="chalking_or_powdering",
            score=chalking_score,
            summary="高亮低饱和区域较多，优先复核粉化、失光或老化粉末化。",
            evidence={
                "chalk_ratio": round(chalk_ratio, 4),
                "low_saturation_ratio": round(low_saturation_ratio, 4),
                "texture_score": round(texture_score, 4),
            },
        ),
        DefectCandidate(
            label="cui_risk_visual_hint",
            score=cui_score,
            summary="存在锈色与暗色异常混合迹象，可作为 CUI 复核线索，不直接作为结论。",
            evidence={
                "rust_ratio": round(rust_ratio, 4),
                "dark_spot_ratio": round(dark_spot_ratio, 4),
                "edge_density": round(edge_density, 4),
            },
        ),
    ]
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    candidates = candidates[:max_candidates]

    return candidates, {
        "rust_ratio": round(rust_ratio, 4),
        "chalk_ratio": round(chalk_ratio, 4),
        "bare_metal_ratio": round(bare_metal_ratio, 4),
        "dark_spot_ratio": round(dark_spot_ratio, 4),
        "low_saturation_ratio": round(low_saturation_ratio, 4),
        "edge_density": round(edge_density, 4),
        "texture_score": round(texture_score, 4),
        "sample_size": {"width": int(sample.shape[1]), "height": int(sample.shape[0])},
    }


def map_official_detections_to_project_candidates(
    raw_candidates: list[DefectCandidate],
    rgb: np.ndarray,
    max_candidates: int,
) -> tuple[list[DefectCandidate], dict[str, Any], list[str]]:
    direct_candidates, direct_metrics = _map_direct_defect_labels(raw_candidates, max_candidates)
    if direct_candidates:
        return (
            direct_candidates,
            {
                "semantic_mapping_applied": True,
                "mapping_strategy": "direct_label_map",
                "raw_detection_count": len(raw_candidates),
                **direct_metrics,
            },
            [
                "模型输出已命中缺陷相关标签，当前结果经过项目语义映射后直接输出。",
                "后续可继续沿用这层标签映射结构统一比赛输出口径。",
            ],
        )

    heuristic_candidates, heuristic_metrics = build_heuristic_defect_candidates(rgb, max_candidates)
    raw_labels = [candidate.label for candidate in raw_candidates[:6]]
    for candidate in heuristic_candidates:
        candidate.evidence["mapping_source"] = "heuristic_fallback"
        candidate.evidence["raw_detection_count"] = len(raw_candidates)
        candidate.box = None
        if raw_labels:
            candidate.evidence["raw_detection_labels"] = raw_labels

    fallback_note = (
        "模型当前输出为通用检测标签，未命中缺陷专用类别，已回退到项目缺陷启发式映射。"
        if raw_candidates
        else "模型当前未检出可用目标框，已回退到项目缺陷启发式映射。"
    )
    return (
        heuristic_candidates,
        {
            "semantic_mapping_applied": True,
            "mapping_strategy": "heuristic_fallback",
            "raw_detection_count": len(raw_candidates),
            "raw_top_labels": raw_labels,
            **heuristic_metrics,
        },
        [
            fallback_note,
            "当前仍属于过渡方案；真正的缺陷语义应优先由缺陷专用模型直接输出。",
        ],
    )


def _map_direct_defect_labels(
    raw_candidates: list[DefectCandidate],
    max_candidates: int,
) -> tuple[list[DefectCandidate], dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    mapped_label_hits: list[str] = []

    for candidate in raw_candidates:
        normalized_label = _normalize_label(candidate.label)
        rule = DIRECT_DEFECT_RULES.get(normalized_label)
        if rule is None:
            continue
        mapped_label_hits.append(candidate.label)
        slot = aggregated.setdefault(
            rule.project_label,
            {
                "score": -1.0,
                "summary": rule.summary,
                "box": candidate.box,
                "source_labels": [],
                "source_scores": [],
            },
        )
        slot["source_labels"].append(candidate.label)
        slot["source_scores"].append(
            {
                "label": candidate.label,
                "score": round(float(candidate.score), 4),
            }
        )
        if candidate.score > slot["score"]:
            slot["score"] = float(candidate.score)
            slot["box"] = candidate.box

    mapped_candidates = [
        DefectCandidate(
            label=project_label,
            score=float(payload["score"]),
            summary=str(payload["summary"]),
            box=payload["box"],
            evidence={
                "mapping_source": "direct_label_map",
                "source_labels": _dedupe_preserve_order(payload["source_labels"]),
                "source_scores": payload["source_scores"],
            },
        )
        for project_label, payload in aggregated.items()
    ]
    mapped_candidates = sorted(mapped_candidates, key=lambda item: item.score, reverse=True)
    mapped_candidates = mapped_candidates[:max_candidates]
    return mapped_candidates, {
        "mapped_label_hits": _dedupe_preserve_order(mapped_label_hits),
    }


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
