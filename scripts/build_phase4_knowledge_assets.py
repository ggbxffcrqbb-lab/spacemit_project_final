from __future__ import annotations

import csv
from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
import subprocess
from typing import Iterable
from xml.etree import ElementTree
import zipfile


TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")
SOURCE_ROOT_NAMES = ("phase4_knowledge_sources", "phase4_knowledge_sources_cn")
RAW_OUTPUT_DIR = "phase4_raw"
CARDS_DIRS = {
    "rules": "phase4_cards_rules",
    "sop": "phase4_cards_sop",
    "visual": "phase4_cards_visual",
}


@dataclass(frozen=True)
class SourceDoc:
    source_root: str
    batch_label: str
    domain: str
    kind: str
    title: str
    url: str
    source_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return repo_root().parent


def knowledge_import_root() -> Path:
    return repo_root() / "data" / "knowledge" / "imported"


def load_manifest_rows(source_root: Path) -> dict[str, dict[str, str]]:
    manifest_dir = source_root / "00_manifest"
    rows: dict[str, dict[str, str]] = {}
    if not manifest_dir.exists():
        return rows

    for csv_path in sorted(manifest_dir.glob("*.csv")):
        if "report" in csv_path.stem.lower():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rel_path = row.get("path", "").strip()
                if not rel_path:
                    continue
                key = Path(rel_path).name.lower()
                rows[key] = row
    return rows


def discover_source_docs() -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    ws_root = workspace_root()

    for root_name in SOURCE_ROOT_NAMES:
        source_root = ws_root / "tmp" / root_name
        if not source_root.exists():
            continue

        manifest_rows = load_manifest_rows(source_root)
        batch_label = "cn" if root_name.endswith("_cn") else "global"

        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            if "00_manifest" in path.parts:
                continue
            if path.suffix.lower() not in {".html", ".pdf", ".docx"}:
                continue

            rel_parts = path.relative_to(source_root).parts
            domain_dir = rel_parts[0]
            domain = domain_dir.split("_", 1)[-1]
            kind = rel_parts[1] if len(rel_parts) > 2 else "raw"
            meta = manifest_rows.get(path.name.lower(), {})
            title = meta.get("title", "").strip() or prettify_stem(path.stem)
            url = meta.get("url", "").strip()
            docs.append(
                SourceDoc(
                    source_root=root_name,
                    batch_label=batch_label,
                    domain=domain,
                    kind=kind,
                    title=title,
                    url=url,
                    source_path=path,
                )
            )
    return docs


def prettify_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip()


def read_text_file(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"unable to decode text file: {path}") from last_error
    return path.read_text(encoding="utf-8", errors="ignore")


def read_html_text(path: Path) -> str:
    raw = read_text_file(path)
    raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</p>", "\n", raw)
    raw = re.sub(r"(?i)</div>", "\n", raw)
    raw = re.sub(r"(?i)</li>", "\n", raw)
    text = re.sub(r"(?is)<[^>]+>", " ", raw)
    return unescape(text)


def read_pdf_text(path: Path) -> str:
    pdftotext = shutil_which("pdftotext")
    if pdftotext:
        proc = subprocess.run(
            [pdftotext, str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout

    try:
        from pypdf import PdfReader  # type: ignore
    except ModuleNotFoundError:
        return f"PDF 文本抽取失败：当前环境缺少 pdftotext / pypdf。原始文件：{path}"

    reader = PdfReader(str(path))
    texts = [page.extract_text() or "" for page in reader.pages]
    joined = "\n".join(texts).strip()
    if joined:
        return joined
    return f"PDF 文本抽取为空。原始文件：{path}"


def read_docx_text(path: Path) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
    except KeyError as exc:
        return f"DOCX 缺少 word/document.xml：{path}"
    except zipfile.BadZipFile:
        return f"DOCX 文件损坏：{path}"

    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    for node in root.findall(".//w:p", namespace):
        parts = [text_node.text or "" for text_node in node.findall(".//w:t", namespace)]
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        compact_lines.append(line)
        previous_blank = is_blank
    return "\n".join(compact_lines).strip()


def read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return read_html_text(path)
    if suffix == ".pdf":
        return read_pdf_text(path)
    if suffix == ".docx":
        return read_docx_text(path)
    raise ValueError(f"unsupported source: {path}")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = value.replace(" ", "-")
    value = re.sub(r"[^0-9a-z\u4e00-\u9fff_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    return value[:96] or "doc"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_raw_document_body(doc: SourceDoc, extracted_text: str) -> str:
    source_rel = doc.source_path.relative_to(workspace_root())
    metadata = [
        f"- 批次：{doc.batch_label}",
        f"- 主题域：{doc.domain}",
        f"- 资料类型：{doc.kind}",
        f"- 原始文件：{source_rel}",
        f"- 原始格式：{doc.source_path.suffix.lower()}",
    ]
    if doc.url:
        metadata.append(f"- 来源网址：{doc.url}")

    return (
        f"# {doc.title}\n\n"
        "## 导入元数据\n\n"
        + "\n".join(metadata)
        + "\n\n## 正文\n\n"
        + extracted_text
        + "\n"
    )


def render_card(title: str, aliases: Iterable[str], boundary: str, bullets: list[str], qa: list[str], sources: list[str]) -> str:
    alias_line = "、".join(aliases)
    key_points = "\n".join(f"- {item}" for item in bullets)
    qa_points = "\n".join(f"- {item}" for item in qa)
    source_points = "\n".join(f"- {item}" for item in sources)
    return (
        f"# {title}\n\n"
        f"## 检索别名\n{alias_line}\n\n"
        f"## 使用边界\n{boundary}\n\n"
        f"## 关键判断\n{key_points}\n\n"
        f"## 问答口径建议\n{qa_points}\n\n"
        f"## 资料来源\n{source_points}\n"
    )


def build_cards() -> list[tuple[Path, str]]:
    imported_root = knowledge_import_root()
    cards: list[tuple[Path, str]] = []

    rule_cards = [
        (
            imported_root / CARDS_DIRS["rules"] / "01_油气管道阴极保护与外防腐协同规则卡.md",
            render_card(
                title="油气管道阴极保护与外防腐协同规则卡",
                aliases=["阴极保护", "外防腐", "埋地管道保护", "CP 协同", "防腐层协同"],
                boundary="适合回答油气管道、储运管线和站场钢制管道的协同防腐思路，不替代阴极保护设计计算、测试布点和投运调试。",
                bullets=[
                    "不要把阴极保护和外防腐涂层理解成二选一。公开标准普遍强调它们是协同体系：外防腐负责隔绝环境，阴极保护负责降低腐蚀驱动力。",
                    "如果同一区域反复返锈、补口后仍有异常电位、或防腐层修补后缺陷继续扩展，优先怀疑体系失配，而不是只盯着单次补漆质量。",
                    "埋地管道巡检时，电位异常、防腐层破损、杂散电流、土壤腐蚀性和排水积液应联动判断，不能只看某一项读数。",
                    "阴极保护参数正常并不等于外防腐层一定完好；反过来，外观没有大面积破损也不代表阴保足够。",
                ],
                qa=[
                    "当用户问“阴保正常为什么还返锈”时，优先回答需要同时排查防腐层完整性、接地杂散、电连接状态和局部环境积液。",
                    "当用户问“阴保数值正常但同一位置反复返锈怎么解释”时，优先回答这更像体系失配信号，不能只把注意力放在阴保读数本身。",
                    "当用户问“是先补口还是先测电位”时，优先回答要先确认异常范围、介质环境和阴保状态，再决定是否直接修补或升级复核。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt21448_2017_cathodic_protection_buried_pipeline.md`",
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt21246_2020_cp_parameter_measurement.md`",
                    "`data/knowledge/imported/phase4_raw/global/oil_gas/phmsa_factsheet_external_corrosion.md`",
                    "`data/knowledge/imported/phase4_raw/global/oil_gas/phmsa_factsheet_internal_corrosion.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["rules"] / "02_埋地管道防腐层与修复前置判断规则卡.md",
            render_card(
                title="埋地管道防腐层与修复前置判断规则卡",
                aliases=["聚乙烯防腐层", "FBE 外涂层", "防腐层修补", "补口修复", "埋地管道外涂层", "划伤后周边起翘", "树脂修复相容性"],
                boundary="适合回答埋地钢质管道外防腐层、补口修补和修复前置判断，不替代具体工艺评定和施工质控方案。",
                bullets=[
                    "聚乙烯层、熔结环氧层和三层粉末体系的巡检重点不同，但都要先判断缺陷是否已穿透至基材、是否伴随边缘翘起和层下扩展。",
                    "发现补口周边返锈、划伤区周边起翘或焊缝附近局部腐蚀时，不宜直接按表面缺陷处理，应先区分施工缺陷、附着力失效和环境诱发。",
                    "对树脂基修复或局部补强类方案，前置条件是缺陷边界清楚、基材状态可确认、且修复后与原有防腐体系相容。",
                    "连续出现同类缺陷、同一线路多点异常或高后果区段受损时，应触发升级复核，而不是简单按零散小缺陷处理。",
                ],
                qa=[
                    "当用户问“埋地管道返锈是不是就是补口坏了”时，优先回答还要排查边缘附着力、阴保状态和局部土壤环境。",
                    "当用户问“发现管道防腐层划伤后周边开始起翘先查什么”时，优先回答先查周边旧层附着力、层下扩展和阴保状态，而不是只补伤口。",
                    "当用户问“树脂修复前要不要先确认原防腐层相容性”时，优先回答要确认，不然局部修复可能和原体系不匹配。",
                    "当用户问“能不能直接包覆修补”时，优先回答先确认基材腐蚀是否已发展、缺陷是否稳定以及修补体系兼容性。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt23257_2017_pe_coating_buried_pipeline.md`",
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt39636_2020_fbe_external_coating.md`",
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt37594_2019_uv_resistant_three_layer_fbe.md`",
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt37195_2018_resin_based_pipeline_repair.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["rules"] / "03_腐蚀控制全生命周期升级复核触发点规则卡.md",
            render_card(
                title="腐蚀控制全生命周期升级复核触发点规则卡",
                aliases=["全生命周期", "升级复核", "风险评估", "防腐失效升级", "复核触发点"],
                boundary="用于回答何时应从现场初判升级到系统复核、风险评估或计划性检维修，不替代完整 RBI、完整性评估或停产决策。",
                bullets=[
                    "出现同一部位反复失效、同一设备多点异常、缺陷扩展趋势加快、或伴随介质/工况变化时，应视为升级复核信号。",
                    "高后果区段、海边高盐雾环境、浪溅区、保温层下潮湿区和焊缝/接管过渡区一旦出现异常，复核优先级应上提。",
                    "风险评估不只看当前缺陷尺寸，还要看缺陷位置、增长速度、后果等级、环境暴露和现有防护体系有效性。",
                    "只做表面修补但不校核根因，会导致失效循环重复，尤其是层下腐蚀、附着力失效和 CUI 线索场景。",
                ],
                qa=[
                    "当用户问“暂时不漏要不要停下来复核”时，优先回答要看缺陷是否处于高后果位置、是否呈现扩展性和是否伴随多点异常。",
                    "当用户问“为什么不能一直补漆”时，优先回答全生命周期控制强调根因排查和系统性复核，而不是反复遮盖表象。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt37183_2023_lifecycle_risk_assessment.md`",
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt37190_2023_pipeline_corrosion_lifecycle_general.md`",
                    "`data/knowledge/imported/phase4_raw/cn/oil_gas/gbt37595_2019_corrosion_control_lifecycle_coating.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["rules"] / "04_海工浪溅区与潮差区高腐蚀风险规则卡.md",
            render_card(
                title="海工浪溅区与潮差区高腐蚀风险规则卡",
                aliases=["浪溅区", "潮差区", "海工高腐蚀", "盐雾腐蚀", "海上结构防腐", "浪溅区起泡"],
                boundary="适合回答海洋平台、码头、海上风电和近海钢结构的高腐蚀区段识别，不替代海工专项评估和船级社检验结论。",
                bullets=[
                    "浪溅区和潮差区通常比完全浸没区或干区更高风险，因为盐分沉积、湿干交替、冲刷和机械损伤叠加更明显。",
                    "节点、扶梯、连接板、焊缝、支撑件和边角位置容易发生应力集中、涂层薄弱和局部积液，应提高关注度。",
                    "如果海边设施掉漆伴随局部起泡、锈水回渗和边缘翘起，优先考虑层下腐蚀或附着力失效，而不是简单把它归为日晒老化。",
                    "海上结构只做局部补漆并不总是足够，必要时要联动检查重防腐体系、牺牲阳极或外加电流保护状态。",
                ],
                qa=[
                    "当用户问“浪溅区为什么更危险”时，优先回答它处在高盐、高湿、反复干湿循环和冲刷叠加区。",
                    "当用户问“浪溅区起泡为什么比普通位置更危险”时，优先回答起泡一旦出现在浪溅区，更可能叠加层下腐蚀和失效扩展。",
                    "当用户问“海边支架掉漆先看什么”时，优先回答先看是否在节点、焊缝、边角或积水区，以及是否有层下扩展线索。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_offshore_wind_corrosion_guide_2025.md`",
                    "`data/knowledge/imported/phase4_raw/global/marine/dnv_corrosion_protection_systems.md`",
                    "`data/knowledge/imported/phase4_raw/global/marine/imo_protective_coatings.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["rules"] / "05_海上风电设施防腐检验重点规则卡.md",
            render_card(
                title="海上风电设施防腐检验重点规则卡",
                aliases=["海上风电防腐", "塔筒防腐", "导管架防腐", "海上风电巡检", "CCS 防腐检验"],
                boundary="适用于海上风电塔筒、导管架、海缆附属钢构等设施的防腐问答，不替代船级社正式检验或项目专项方案。",
                bullets=[
                    "海上风电设施应按部位分层看：浪溅区、潮差区、飞溅区、密闭腔体、法兰连接区、登乘与维护区域的失效模式不同。",
                    "塔筒门口、平台通道、螺栓连接、焊缝热影响区和检修高频触达区容易兼有机械损伤与腐蚀扩展。",
                    "如果现场问“只见局部锈迹是否可以拖到下次窗口”，应先判断是否位于高后果部位、是否影响密封排水和是否已进入层下扩展。",
                    "征求意见稿、正式指南和编制说明一起看时，更容易把检验重点、证据留存和整改闭环要求串起来。",
                ],
                qa=[
                    "当用户问“塔筒掉漆是不是正常老化”时，优先回答要先分区段，再看是否伴随机械磕碰、边缘起翘和盐雾积聚。",
                    "当用户问“海上风电防腐检查看哪几块”时，优先回答先看浪溅区、连接区、焊缝、平台通道和排水密封薄弱部位。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_offshore_wind_corrosion_guide_2025.md`",
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_offshore_wind_corrosion_draft_2024.md`",
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_offshore_wind_corrosion_draft_explainer_2024.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["rules"] / "06_水下生产系统与海工发证防腐证据链规则卡.md",
            render_card(
                title="水下生产系统与海工发证防腐证据链规则卡",
                aliases=["水下生产系统", "海工发证", "防腐证据链", "海工检验资料", "发证检查"],
                boundary="用于回答海工发证前需要看哪些防腐证据与资料链，不替代正式发证审图和见证程序。",
                bullets=[
                    "海工发证场景下，防腐不仅看外观，还要看材料适用性、表面处理记录、施工工艺、检验记录和缺陷处置闭环。",
                    "如果某部件处于难以二次进入或水下服役环境，应提高对涂层完整性、阴保设计和前期证据留存的要求。",
                    "现场看起来“问题不大”并不能替代文件证据。对发证类问题，资料完整性和可追溯性本身就是判断要点。",
                    "一旦发现维修、返工、局部补涂或材料替换，应追问变更记录是否闭环，避免只依据外观做乐观结论。",
                ],
                qa=[
                    "当用户问“水下系统发证前主要看什么”时，优先回答看材料、防护体系、施工检验记录和变更处置证据。",
                    "当用户问“外观没问题是不是就能过”时，优先回答发证类问题还必须看工艺和检验文件链。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_subsea_system_certification_2016.md`",
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_subsea_system_certification_2016-pdf.md`",
                ],
            ),
        ),
    ]

    sop_cards = [
        (
            imported_root / CARDS_DIRS["sop"] / "01_有限空间作业前置检查SOP卡.md",
            render_card(
                title="有限空间作业前置检查 SOP 卡",
                aliases=["有限空间", "入罐前检查", "进仓前检查", "受限空间", "入内作业", "进去拍照", "待几分钟"],
                boundary="适合回答受限空间进入前的前置动作和常见漏项，不替代企业票证、监护和救援专项方案。",
                bullets=[
                    "先确认作业审批、隔离、置换、通风、气体检测、持续监测、监护和应急救援准备是否齐全，再讨论进不进去。",
                    "只进去几分钟拍照、看一眼、拿个工具，也不能跳过气体检测和监护要求。",
                    "涉及腐蚀检维修时，还要额外关注残液、沉积物、盲区积气、清罐死角和临时拆保温后的环境变化。",
                    "一旦存在可燃、有毒、缺氧、隔离不到位或救援条件不足，不应把现场工作压缩成“快速进去一下”。",
                ],
                qa=[
                    "当用户问“只进去待几分钟可不可以不测”时，优先回答时间短不是豁免条件。",
                    "当用户问“只进去拍个照也要做受限空间检测吗”时，优先回答拍照、观察、取物这类短时进入同样不能跳过检测和监护。",
                    "当用户问“先做哪一步”时，优先回答先做隔离、通风和检测，再看监护与救援准备。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/sop_safety/gb46768_2025_confined_space_safety.md`",
                    "`data/knowledge/imported/phase4_raw/cn/sop_safety/mem_confined_space_safe_rescue_2023.md`",
                    "`data/knowledge/imported/phase4_raw/global/sop_safety/osha3639_ventilation_shipyard.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["sop"] / "02_动火与受限空间隔离检测复测SOP卡.md",
            render_card(
                title="动火与受限空间隔离检测复测 SOP 卡",
                aliases=["动火", "热作业", "隔离", "气体分析", "复测"],
                boundary="适合回答动火、受限空间和介质相关特殊作业的隔离检测流程，不替代正式票证制度和现场许可审批。",
                bullets=[
                    "动火前不只看是否清理干净，还要确认介质隔离、盲板或断开状态、置换通风和气体分析是否到位。",
                    "可燃、有毒、缺氧风险不是一次检测就结束。作业条件变化、停工再开、通风中断或人员轮换时，应考虑复测和持续监测。",
                    "带压、残液、连通管线、隐蔽空间和邻近设备返气，是现场最容易漏掉的复合风险点。",
                    "如果场景同时涉及防腐检修、焊补、切割、拆保温或开盖，优先按高风险组合场景口径回答，而不是按单一动作回答。",
                ],
                qa=[
                    "当用户问“动火前一定要测气吗”时，优先回答涉及可燃、有毒或受限空间时不应跳过。",
                    "当用户问“刚测过还要不要再测”时，优先回答作业条件改变就要重新确认，不宜机械套用一次结果。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/sop_safety/gb30871_2022_special_operations_safety.md`",
                    "`data/knowledge/imported/phase4_raw/global/sop_safety/osha3732_hot_work_marine_terminals.md`",
                    "`data/knowledge/imported/phase4_raw/global/sop_safety/osha_fs3586_hot_work_shipyards.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["sop"] / "03_海工平台搁置期间防腐检查SOP卡.md",
            render_card(
                title="海工平台搁置期间防腐检查 SOP 卡",
                aliases=["搁置检验", "海工平台搁置", "停用平台检查", "封存平台防腐"],
                boundary="适合回答海上移动平台、停用或搁置状态设施的防腐检查流程，不替代船级社检验计划和复启程序。",
                bullets=[
                    "搁置状态不等于风险静止。长期停用期间应关注排水、封闭空间通风、潮湿积液、阴保状态和暴露表面退化。",
                    "检查顺序建议先看高后果暴露部位，再看密闭空间、登靠设施、边角焊缝和维护通道。",
                    "如果平台准备复启，应把搁置期防腐状态、维修记录和新增缺陷一起纳入复启前证据链。",
                    "发现局部异常时，不要只记录“有锈蚀”，还应记录区段、范围、失效形态和是否伴随涂层起泡、脱落或密封失效。",
                ],
                qa=[
                    "当用户问“平台停着不动还要看防腐吗”时，优先回答搁置期同样会累积潮湿、盐雾和排水失效风险。",
                    "当用户问“复启前先看哪里”时，优先回答先看高暴露区、密闭腔体和证据留存是否完整。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_laidup_mobile_platform_2022.md`",
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_laidup_mobile_platform_2022-pdf.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["sop"] / "04_海上风电设施防腐检验现场流程SOP卡.md",
            render_card(
                title="海上风电设施防腐检验现场流程 SOP 卡",
                aliases=["海上风电检验", "防腐现场流程", "塔筒巡检流程", "导管架检查流程"],
                boundary="适合回答海上风电设施现场检验流程、记录重点和整改闭环，不替代正式检验计划和海上作业许可。",
                bullets=[
                    "现场流程建议按“分区识别、缺陷拍照、范围记录、风险初判、升级触发、整改闭环”推进，不要只停留在拍照留痕。",
                    "浪溅区、平台通道、焊缝、法兰连接、门口和高频检修部位应优先覆盖。",
                    "缺陷记录除了位置，还应写清缺陷形态、边界、是否层下扩展、是否影响排水密封和是否已接近结构关键区。",
                    "如果同一风机多个部位同时异常，或同类部件批量出现类似失效，应考虑从单点缺陷升级到体系性问题复核。",
                ],
                qa=[
                    "当用户问“巡检记录要写什么”时，优先回答位置、分区、缺陷形态、范围、风险级别和升级建议缺一不可。",
                    "当用户问“看到一点锈先补还是先复核”时，优先回答要先看所在分区和是否伴随边缘起翘、层下扩展。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_offshore_wind_corrosion_guide_2025.md`",
                    "`data/knowledge/imported/phase4_raw/cn/marine/ccs_offshore_wind_corrosion_draft_2024.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["sop"] / "05_防腐修复前表面处理与附着力复核SOP卡.md",
            render_card(
                title="防腐修复前表面处理与附着力复核 SOP 卡",
                aliases=["表面处理", "附着力复核", "划格", "拉开法", "修补前确认"],
                boundary="适合回答防腐修补前的表面处理和附着力复核逻辑，不替代产品施工卡、工艺评定和验收规范。",
                bullets=[
                    "修补前先确认缺陷是不是停留在表层，还是已经进入层下扩展、附着力失效或基材腐蚀阶段。",
                    "如果边缘起翘、划伤周边成片脱层、拉开法明显失效或粉化严重，应慎用只补面漆的方案。",
                    "表面处理质量直接影响修补效果。污染物、盐分、潮湿和旧涂层松动未处理干净时，返修寿命会明显受限。",
                    "附着力复核宜与缺陷分级、环境条件和后续防护体系一起看，不能孤立看某一次数值或某一格结果。",
                ],
                qa=[
                    "当用户问“先补漆还是先做附着力”时，优先回答先分清缺陷性质和边界，再决定是否进入修补。",
                    "当用户问“粉化但没掉皮能不能直接压一层”时，优先回答先看旧层强度和表面处理是否满足前提。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt31586_2_2015_cross_cut_acceptance.md`",
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt5210_2006_pull_off_adhesion.md`",
                    "`data/knowledge/imported/phase4_raw/global/visual_defects/hempel_surface_preparation.md`",
                ],
            ),
        ),
    ]

    visual_cards = [
        (
            imported_root / CARDS_DIRS["visual"] / "01_涂层生锈等级视觉判读卡.md",
            render_card(
                title="涂层生锈等级视觉判读卡",
                aliases=["生锈等级", "返锈判读", "锈点", "锈斑", "生锈评级"],
                boundary="适合回答视觉巡检中锈蚀分布和等级趋势，不替代正式评级判定表和结构完整性评估。",
                bullets=[
                    "视觉上先看锈蚀是点状、片状还是沿边缘扩展，再看是否有锈水回渗、起泡或划伤关联线索。",
                    "零星锈点和成片返锈的处理优先级不同；若锈蚀集中在焊缝、边角、接管根部和积水区，应提高警惕。",
                    "若表面锈迹伴随涂层鼓包、边缘翘起或明显层下扩展，不能只按轻微外观老化处理。",
                    "同一设备多点出现类似锈蚀形态时，往往提示体系性问题而不只是单点损伤。",
                ],
                qa=[
                    "当用户问“这是锈点还是大问题”时，优先回答看分布、位置和是否伴随层下扩展迹象。",
                    "当用户问“焊缝边一点返锈要不要紧”时，优先回答焊缝和边角位属于优先复核部位。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt30789_3_2014_rusting_degree.md`",
                    "`data/knowledge/imported/phase4_raw/global/visual_defects/jotun_common_building_defects.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["visual"] / "02_剥落起皮与边缘翘起视觉判读卡.md",
            render_card(
                title="剥落起皮与边缘翘起视觉判读卡",
                aliases=["剥落", "起皮", "翘边", "脱层", "边缘起翘"],
                boundary="适合回答片状剥落、边缘起翘和层下扩展的视觉判读，不替代附着力试验和返修工艺决策。",
                bullets=[
                    "看见剥落时，先分辨是点状脱落、片状起皮，还是沿划伤、焊缝、切边向周围扩展。",
                    "边缘翘起通常比单纯表面掉色更值得警惕，因为它更容易提示附着力下降或层下腐蚀已经启动。",
                    "如果剥落边界下方已有锈迹、潮湿、盐分残留或旧层粉化，简单覆盖往往不能解决根因。",
                    "局部剥落发生在浪溅区、保温切口、法兰边和焊缝热影响区时，应优先排查环境与工况诱因。",
                ],
                qa=[
                    "当用户问“这是脱层还是磕碰掉漆”时，优先回答看边界是否翘起、周边是否继续扩展。",
                    "当用户问“补一下边就行吗”时，优先回答先确认旧层是否已经失去附着力。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt30789_5_2015_flaking_degree.md`",
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt30789_8_2015_delamination_near_scribe.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["visual"] / "03_粉化与老化失效视觉判读卡.md",
            render_card(
                title="粉化与老化失效视觉判读卡",
                aliases=["粉化", "老化失效", "表层发粉", "掉粉", "涂层老化"],
                boundary="适合回答表层老化、掉粉和继续服役风险，不替代涂层寿命评估和返修工艺试验。",
                bullets=[
                    "粉化常表现为手擦掉粉、表面失光、颜色发灰发白，但不一定已经发展到剥落阶段。",
                    "如果粉化严重且伴有底层松动、局部裂纹或附着力下降，不宜只把它当作单纯外观老化。",
                    "海边高紫外、盐雾和高温潮湿循环环境，会放大粉化后的失效扩展速度。",
                    "粉化区返修前尤其要关注旧涂层残留强度和表面处理是否到位。",
                ],
                qa=[
                    "当用户问“粉化但没掉皮是不是还能拖”时，优先回答要看是否已影响后续附着力和保护能力。",
                    "当用户问“直接压一层面漆行不行”时，优先回答先确认旧层是否还能作为可靠基底。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt30789_6_2015_chalking_degree.md`",
                    "`data/knowledge/imported/phase4_raw/global/visual_defects/hempel_explanatory_notes_pds.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["visual"] / "04_划线周边剥离与附着力失效视觉判读卡.md",
            render_card(
                title="划线周边剥离与附着力失效视觉判读卡",
                aliases=["划线周边剥离", "划伤边缘翘起", "附着力失效", "沿划痕脱层"],
                boundary="适合回答人工缺陷、划痕、碰伤周边的视觉演变，不替代正式附着力试验和实验室判断。",
                bullets=[
                    "划线或碰伤周边的剥离如果持续向外扩展，往往说明问题不只在伤口本身，而是周围涂层粘附或界面稳定性不足。",
                    "现场要区分单纯机械损伤和“损伤后继续层下扩展”的状态，后者更需要升级处理。",
                    "如果同类划伤在多处都表现出类似翘边，优先怀疑体系性附着力不足、表面处理不良或环境介入。",
                    "焊缝附近、边角和高盐分沉积区的划伤周边剥离风险通常更高。",
                ],
                qa=[
                    "当用户问“划痕周边起边是不是没事”时，优先回答要看是否正在向周边扩展。",
                    "当用户问“是不是只把伤口补上就行”时，优先回答先确认周边旧层是否还牢靠。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt30789_8_2015_delamination_near_scribe.md`",
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt31586_2_2015_cross_cut_acceptance.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["visual"] / "05_划格与拉开法结果现场解释卡.md",
            render_card(
                title="划格与拉开法结果现场解释卡",
                aliases=["划格试验", "拉开法", "附着力结果", "验收线索", "现场解释"],
                boundary="适合回答附着力试验结果如何在现场口径中解释，不替代正式验收判级和产品说明书要求。",
                bullets=[
                    "划格和拉开法是帮助判断旧层是否还能继续作为基底的证据，不是独立于现场环境和缺陷形态的单一结论。",
                    "如果视觉上已经有大面积翘边、粉化、层下腐蚀，试验结果再好看也要谨慎解释样点代表性。",
                    "若试验失败集中在界面而不是涂层内部，通常更要关注表面处理、污染物和旧层老化问题。",
                    "试验结果应用于现场时，应同时记录取样位置、缺陷背景和环境条件，避免脱离场景引用。",
                ],
                qa=[
                    "当用户问“拉开法通过是不是就能直接补”时，优先回答还要看缺陷分布、样点代表性和环境条件。",
                    "当用户问“划格差一点要不要全做”时，优先回答看缺陷是否局部、是否有扩展趋势以及是否处于关键位置。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt31586_2_2015_cross_cut_acceptance.md`",
                    "`data/knowledge/imported/phase4_raw/cn/visual_defects/gbt5210_2006_pull_off_adhesion.md`",
                    "`data/knowledge/imported/phase4_raw/global/visual_defects/jotun_penguard_hb_application_guide.md`",
                ],
            ),
        ),
        (
            imported_root / CARDS_DIRS["visual"] / "06_CUI线索与保温层下腐蚀视觉判读卡.md",
            render_card(
                title="CUI 线索与保温层下腐蚀视觉判读卡",
                aliases=["CUI", "保温层下腐蚀", "包覆鼓包", "保温潮湿", "外壁渗水", "海边管廊支架返锈"],
                boundary="适合回答保温层下腐蚀的外观线索和现场优先排查点，不替代拆检、测厚和完整性评估。",
                bullets=[
                    "保温层外表面看似完整，并不代表内部没有问题。鼓包、渗水、接缝潮湿、包覆变形和锈水回渗都是典型线索。",
                    "温差区、低点、支架穿透点、法兰阀门、保温切口和长期积水区域，是 CUI 更应优先排查的部位。",
                    "如果海边管廊或高湿区同时出现返锈和保温潮湿，应把它视为高优先级复核信号。",
                    "视觉线索只能作为拆检触发依据之一，不能替代厚度确认和范围判断。",
                ],
                qa=[
                    "当用户问“保温外面没漏是不是不用拆”时，优先回答外观平静并不能排除内部潮湿和层下腐蚀。",
                    "当用户问“海边管廊支架返锈还伴随保温潮湿先查哪里”时，优先回答先查低点、支架穿透点、接缝和法兰阀门过渡区。",
                    "当用户问“先拆哪块”时，优先回答先看低点、支架、接缝和法兰阀门过渡区。",
                ],
                sources=[
                    "`data/knowledge/imported/phase4_raw/global/marine/hse_cui_plant_pipework.md`",
                    "`data/knowledge/imported/phase4_raw/global/oil_gas/iogp_jip33_insulation_piping_equipment_s738.md`",
                ],
            ),
        ),
    ]

    cards.extend(rule_cards)
    cards.extend(sop_cards)
    cards.extend(visual_cards)
    return cards


def shutil_which(executable: str) -> str | None:
    try:
        from shutil import which
    except ImportError:
        return None
    return which(executable)


def build_raw_documents(docs: list[SourceDoc]) -> list[Path]:
    output_paths: list[Path] = []
    imported_root = knowledge_import_root()

    for doc in docs:
        extracted = normalize_text(read_source_text(doc.source_path))
        if not extracted:
            extracted = f"未能抽取到正文内容。原始文件：{doc.source_path}"

        base_stem = doc.source_path.stem
        if doc.source_path.suffix.lower() == ".pdf":
            base_stem = f"{base_stem}-pdf"
        elif doc.source_path.suffix.lower() == ".docx":
            base_stem = f"{base_stem}-docx"

        output_path = (
            imported_root
            / RAW_OUTPUT_DIR
            / doc.batch_label
            / doc.domain
            / f"{slugify(base_stem)}.md"
        )
        body = build_raw_document_body(doc, extracted)
        write_text(output_path, body)
        output_paths.append(output_path)

    return output_paths


def build_card_documents() -> list[Path]:
    output_paths: list[Path] = []
    for path, content in build_cards():
        write_text(path, content)
        output_paths.append(path)
    return output_paths


def main() -> None:
    docs = discover_source_docs()
    raw_paths = build_raw_documents(docs)
    card_paths = build_card_documents()
    print(f"phase4 raw docs: {len(raw_paths)}")
    print(f"phase4 cards: {len(card_paths)}")


if __name__ == "__main__":
    main()
