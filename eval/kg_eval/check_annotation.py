#!/usr/bin/env python3
"""
標註一致性與完整性檢查工具

檢查 ground truth JSON 標註的一致性與完整性問題。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def check_annotation_consistency(ground_truth: Dict[str, Any]) -> List[str]:
    """
    檢查標註一致性

    Args:
        ground_truth: Ground truth JSON 字典，包含 "entities" 和 "relations"

    Returns:
        問題清單（空清單表示無問題）
    """
    issues = []

    entities = ground_truth.get("entities", {})
    relations = ground_truth.get("relations", {})

    # 檢查關係中的實體是否存在
    for relation_key, relation_data in relations.items():
        head = relation_data.get("head")
        tail = relation_data.get("tail")

        if not head:
            issues.append(f"關係 {relation_key} 的 head 為空")
        elif head not in entities:
            issues.append(f"關係 {relation_key} 的 head 實體 '{head}' 不存在於 entities")

        if not tail:
            issues.append(f"關係 {relation_key} 的 tail 為空")
        elif tail not in entities:
            issues.append(f"關係 {relation_key} 的 tail 實體 '{tail}' 不存在於 entities")

    # 檢查重複實體（理論上字典 key 不會重複，但檢查一下結構）
    entity_names = list(entities.keys())
    if len(entity_names) != len(set(entity_names)):
        issues.append("存在重複的實體名稱（字典 key 重複）")

    # 檢查實體結構完整性
    for entity_name, entity_data in entities.items():
        if not isinstance(entity_data, dict):
            issues.append(f"實體 '{entity_name}' 的資料格式不正確（應為字典）")
        else:
            if "entity_type" not in entity_data:
                issues.append(f"實體 '{entity_name}' 缺少 entity_type 欄位")

    # 檢查關係結構完整性
    for relation_key, relation_data in relations.items():
        if not isinstance(relation_data, dict):
            issues.append(f"關係 {relation_key} 的資料格式不正確（應為字典）")
        else:
            required_fields = ["head", "tail", "relation_type"]
            for field in required_fields:
                if field not in relation_data:
                    issues.append(f"關係 {relation_key} 缺少 {field} 欄位")

    return issues


def check_annotation_completeness(
    ground_truth: Dict[str, Any], document_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    檢查標註完整性

    Args:
        ground_truth: Ground truth JSON 字典
        document_text: 原始文件文字（可選，用於更深入的檢查）

    Returns:
        統計資訊字典
    """
    entities = ground_truth.get("entities", {})
    relations = ground_truth.get("relations", {})

    stats = {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "entity_types": {},
        "relation_types": {},
    }

    # 統計實體類型
    for entity_data in entities.values():
        entity_type = entity_data.get("entity_type", "UNKNOWN")
        stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1

    # 統計關係類型
    for relation_data in relations.values():
        relation_type = relation_data.get("relation_type", "UNKNOWN")
        stats["relation_types"][relation_type] = stats["relation_types"].get(relation_type, 0) + 1

    return stats


def main():
    ap = argparse.ArgumentParser(
        description="檢查 ground truth JSON 標註的一致性與完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 檢查標註一致性
  python check_annotation.py --input eval/kg_eval/text-02_ground_truth.json

  # 同時檢查一致性和完整性
  python check_annotation.py --input eval/kg_eval/text-02_ground_truth.json --completeness
        """,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Ground truth JSON 檔案路徑",
    )
    ap.add_argument(
        "--completeness",
        action="store_true",
        help="同時檢查標註完整性（統計資訊）",
    )
    ap.add_argument(
        "--document",
        help="原始文件文字檔案路徑（用於完整性檢查，可選）",
    )
    args = ap.parse_args()

    # 讀取 ground truth JSON
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"錯誤：檔案不存在：{input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
    except json.JSONDecodeError as e:
        print(f"錯誤：JSON 格式錯誤：{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"錯誤：讀取檔案失敗：{e}", file=sys.stderr)
        sys.exit(1)

    # 檢查一致性
    print(f"檢查標註一致性：{input_path}")
    print("-" * 60)
    issues = check_annotation_consistency(ground_truth)

    if issues:
        print("發現標註問題：")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print(f"\n總共發現 {len(issues)} 個問題")
        sys.exit(1)
    else:
        print("[OK] 標註一致性檢查通過")

    # 檢查完整性（如果指定）
    if args.completeness:
        print("\n" + "-" * 60)
        print("檢查標註完整性")
        print("-" * 60)

        document_text = None
        if args.document:
            doc_path = Path(args.document)
            if doc_path.exists():
                document_text = doc_path.read_text(encoding="utf-8")
            else:
                print(f"警告：文件檔案不存在：{doc_path}，跳過文件相關檢查", file=sys.stderr)

        stats = check_annotation_completeness(ground_truth, document_text)

        print(f"已標註實體數：{stats['entity_count']}")
        print(f"已標註關係數：{stats['relation_count']}")

        if stats["entity_types"]:
            print("\n實體類型分布：")
            for entity_type, count in sorted(stats["entity_types"].items()):
                print(f"  {entity_type}: {count}")

        if stats["relation_types"]:
            print("\n關係類型分布：")
            for relation_type, count in sorted(stats["relation_types"].items()):
                print(f"  {relation_type}: {count}")

    print("\n檢查完成")


if __name__ == "__main__":
    main()
