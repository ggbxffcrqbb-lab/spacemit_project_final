from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import load_app_config
from app.rag import LocalKnowledgeBase
from app.rag.query_guard import contains_domain_signal, filter_strong_hits


POSITIVE_QUERIES = [
    "涂层起泡怎么判断先补漆还是先复核",
    "发现裂纹能不能先补漆遮住",
    "储罐外壁腐蚀如何初判风险",
    "保温层下疑似腐蚀怎么处理",
    "进入罐内前需要检查什么",
    "进入有限空间只待几分钟可以吗",
    "动火前要确认哪些风险",
    "在封闭支撑腿上直接焊一下行不行",
    "高风险场景怎么先做保守处置",
    "什么时候不能只做表面补漆",
    "阴极保护和涂层哪个更重要",
    "储罐巡检应该看哪些地方",
    "巡检记录要写什么",
    "发现锈水回渗说明什么",
    "暂时不漏还要不要管",
    "储罐底板腐蚀能不能直接补涂",
    "有限空间事故后先做什么",
    "有限空间施救能不能盲目下去救人",
    "有限空间作业前要准备哪些应急内容",
    "巡检记录除了位置还要写什么",
    "同一储罐多处异常还算不算单点问题",
    "埋地管道电位异常先看防腐层还是先看阴保",
    "埋地管道返锈是不是一定代表补口坏了",
    "阴保数值正常但同一位置反复返锈怎么解释",
    "发现管道防腐层划伤后周边开始起翘先查什么",
    "树脂修复前要不要先确认原防腐层相容性",
    "海边管廊支架返锈还伴随保温潮湿先查哪里",
    "保温外面没漏是不是就不用拆开看",
    "浪溅区起泡为什么比普通位置更危险",
    "海上风电塔筒门口掉漆先看什么",
    "海上风电平台通道边角返锈算不算高风险",
    "海工平台搁置几年后复启前防腐先看哪几块",
    "水下生产系统发证前防腐一般看哪些证据",
    "看到一点锈迹能不能拖到下次海上窗口再处理",
    "粉化严重但还没剥落能不能直接压一层面漆",
    "划线周边翘边算不算附着力失效信号",
    "划格试验一般是拿来判断什么的",
    "拉开法通过是不是就能直接补漆",
    "焊缝附近局部返锈为什么要优先复核",
    "动火前刚测过气还要不要复测",
    "只进去拍个照也要做受限空间检测吗",
    "发现多点异常什么时候要升级到系统性复核",
    "埋地管道防腐层破了是不是一定要马上开挖",
    "阴极保护电位偏低先查电源还是先查涂层",
    "管道支架根部反复返锈一般先怀疑什么",
    "法兰附近老是有锈迹先看密封还是先看防腐",
    "保温层接口发黑发潮是不是要怀疑 CUI",
    "输油管线外壁局部鼓包先停运还是先复核",
    "发现杂散电流干扰时现场先做什么",
    "阀门底部和低点部位为什么更容易先出问题",
    "浪溅区起泡为什么比普通外壁更危险",
    "海上平台栏杆焊缝边缘返锈先看哪一项",
    "盐雾环境下掉漆很快通常说明什么",
    "海工钢结构只补面漆为什么往往不够",
    "海边支架有保温又返锈先拆保温还是先拍照记录",
    "牺牲阳极看着还在为什么结构还是会锈",
    "海上风电塔筒门口掉漆为什么要优先复核",
    "潮差区和全浸区的腐蚀关注点有什么不同",
    "这像起泡还是像附着力失效",
    "锈线从焊缝边上出来一般说明什么",
]

NEGATIVE_QUERIES = [
    "今天天气怎么样",
    "帮我写一封请假邮件",
    "这个项目的git怎么提交",
    "摄像头分辨率是多少",
    "怎么做红烧肉",
    "帮我写个自我介绍",
]


def build_knowledge_base() -> LocalKnowledgeBase:
    config = load_app_config(PROJECT_ROOT / "configs" / "voice.yaml")

    # Windows 侧离线补库时，强制使用当前工作区知识库与本地索引，
    # 避免误读板端路径缓存。
    if sys.platform.startswith("win"):
        config.rag.knowledge_dir = PROJECT_ROOT / "data" / "knowledge"
        config.rag.index_path = PROJECT_ROOT / "data" / "index" / "knowledge_index.local.json"

    return LocalKnowledgeBase(config.rag)


def main():
    kb = build_knowledge_base()

    hit_count = 0
    for index, query in enumerate(POSITIVE_QUERIES, start=1):
        hits = kb.search(query, top_k=3)
        if hits:
            hit_count += 1
            top = hits[0]
            print(
                f"[{index:02d}] HIT  | {query} | {top.title} | {Path(top.source_path).name} | score={top.score}"
            )
        else:
            print(f"[{index:02d}] MISS | {query}")

    negative_pass_count = 0
    base_index = len(POSITIVE_QUERIES)
    for offset, query in enumerate(NEGATIVE_QUERIES, start=1):
        hits = kb.search(query, top_k=3)
        strong_hits = filter_strong_hits(query, hits)
        if not contains_domain_signal(query) and not strong_hits:
            negative_pass_count += 1
            print(f"[{base_index + offset:02d}] PASS | {query} | fallback_expected")
        else:
            top = strong_hits[0] if strong_hits else hits[0]
            print(
                f"[{base_index + offset:02d}] FAIL | {query} | "
                f"{top.title} | {Path(top.source_path).name} | score={top.score}"
            )

    print(
        f"summary: positive={hit_count}/{len(POSITIVE_QUERIES)} hits | "
        f"negative={negative_pass_count}/{len(NEGATIVE_QUERIES)} guarded"
    )


if __name__ == "__main__":
    main()
