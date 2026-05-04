"""在带 RDKit 的外部环境中运行的辅助脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cleanup_and_fail(output_file: Path, message: str) -> int:
    """清理不完整输出并把失败原因写到标准错误。"""
    if output_file.exists():
        output_file.unlink()
    print(message, file=sys.stderr)
    return 1


def sdf_to_smiles(input_sdf: Path, output_smi: Path, allow_invalid_records: bool = False) -> int:
    """把 SDF 转成 smiles-tab-name 文件。"""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False)
    output_smi.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_smi.open("w", encoding="utf-8") as handle:
        for index, mol in enumerate(supplier, start=1):
            if mol is None:
                if allow_invalid_records:
                    print(f"SDF 第 {index} 条记录解析失败，已跳过。", file=sys.stderr)
                    continue
                return _cleanup_and_fail(output_smi, f"SDF 第 {index} 条记录解析失败。")
            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            if not name:
                name = f"mol_{index:04d}"
            handle.write(f"{smiles}\t{name}\n")
            written += 1
    if written == 0:
        return _cleanup_and_fail(output_smi, "SDF 中没有可用分子。")
    return 0


def normalize_smiles_file(input_file: Path, output_smi: Path, allow_invalid_records: bool = False) -> int:
    """把文本候选文件统一整理成 smiles-tab-name 文件。"""
    from rdkit import Chem

    try:
        from rdkit import RDLogger
    except ImportError:
        RDLogger = None

    output_smi.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    if RDLogger is not None:
        RDLogger.DisableLog("rdApp.error")
    try:
        with input_file.open("r", encoding="utf-8") as source, output_smi.open("w", encoding="utf-8") as target:
            for index, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
                first = parts[0].strip() if len(parts) > 0 else ""
                second = parts[1].strip() if len(parts) > 1 else ""

                first_valid = bool(first) and Chem.MolFromSmiles(first) is not None
                second_valid = bool(second) and Chem.MolFromSmiles(second) is not None

                if first_valid:
                    smiles = first
                    name = second or f"mol_{index:04d}"
                elif second_valid:
                    smiles = second
                    name = first or f"mol_{index:04d}"
                elif allow_invalid_records:
                    print(f"第 {index} 行不是有效的 SMILES 记录，已跳过。", file=sys.stderr)
                    continue
                else:
                    return _cleanup_and_fail(output_smi, f"第 {index} 行不是有效的 SMILES 记录。")

                target.write(f"{smiles}\t{name}\n")
                written += 1
    finally:
        if RDLogger is not None:
            RDLogger.EnableLog("rdApp.error")
    if written == 0:
        return _cleanup_and_fail(output_smi, "文本文件中没有可用分子。")
    return 0


def main() -> int:
    """执行 RDKit 辅助命令。"""
    parser = argparse.ArgumentParser(description="TargetMol RDKit helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sdf_parser = subparsers.add_parser("sdf-to-smiles")
    sdf_parser.add_argument("--input-sdf", required=True)
    sdf_parser.add_argument("--output-smi", required=True)
    sdf_parser.add_argument("--allow-invalid-records", action="store_true")

    normalize_parser = subparsers.add_parser("normalize-smiles-file")
    normalize_parser.add_argument("--input-file", required=True)
    normalize_parser.add_argument("--output-smi", required=True)
    normalize_parser.add_argument("--allow-invalid-records", action="store_true")

    args = parser.parse_args()
    if args.command == "sdf-to-smiles":
        return sdf_to_smiles(
            Path(args.input_sdf),
            Path(args.output_smi),
            allow_invalid_records=args.allow_invalid_records,
        )
    if args.command == "normalize-smiles-file":
        return normalize_smiles_file(
            Path(args.input_file),
            Path(args.output_smi),
            allow_invalid_records=args.allow_invalid_records,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
