#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path
from random import randint
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai_client.openai_client import MyOpenAIClient
from qa_item import QAItem


def _read_file_content(file_path: str) -> str:
    try:
        with open(file_path) as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"File {file_path} is empty.")
            return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise


def _pick_version() -> str:
    prompts_dir = PROJECT_ROOT / "prompts"
    versions = sorted(p.name for p in prompts_dir.iterdir() if p.is_dir())
    if not versions:
        raise RuntimeError(f"No version folders found in {prompts_dir}")
    if len(versions) == 1:
        print(f"Using prompt version: {versions[0]}")
        return versions[0]
    print("Available prompt versions:")
    for i, v in enumerate(versions, start=1):
        print(f"  {i}) {v}")
    while True:
        choice = input(f"Select version (1-{len(versions)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(versions):
            return versions[int(choice) - 1]
        print(f"  Please enter a number between 1 and {len(versions)}.")



_VAGUE_PHRASES = {"be careful", "good luck", "take your time", "stay safe", "use caution"}

def dim_sanity_check(qa_item: QAItem) -> None:
    problems = []
    if len(qa_item.safety_info) < 80:
        problems.append(f"safety_info too short ({len(qa_item.safety_info)} chars, min 80)")
    if not qa_item.steps:
        problems.append("steps list is empty")
    if not qa_item.tools_required:
        problems.append("tools_required list is empty")
    _UNREALISTIC_TOOL_PHRASES = {"professional-grade", "trade-only"}
    for tool in qa_item.tools_required:
        if len(tool.strip()) < 3:  # passes for axe, saw, bit, hoe, etc.
            problems.append(f"tool name too short: '{tool}'")
        for phrase in _UNREALISTIC_TOOL_PHRASES:
            if phrase in tool.strip().lower():
                problems.append(f"unrealistic tool detected ('{phrase}'): '{tool}'")

    if not qa_item.tips:
        problems.append("tips list is empty")
    for tip in qa_item.tips:
        for phrase in _VAGUE_PHRASES:
            if phrase in tip.strip().lower():
                problems.append(f"vague tip detected ('{phrase}'): '{tip}'")
    for phrase in _VAGUE_PHRASES:
        if phrase in qa_item.safety_info.strip().lower():
            problems.append(f"vague safety phrase detected ('{phrase}'): '{qa_item.safety_info}'")
    if problems:
        raise ValueError("Sanity check failed:\n" + "\n".join(f"  - {p}" for p in problems))


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.92


def batch_dedup_check(qa_folder: Path, client: MyOpenAIClient) -> None:
    qa_files = sorted(qa_folder.glob("*.qa"))
    if len(qa_files) < 2:
        return

    items = []
    for f in qa_files:
        try:
            items.append((f, QAItem.model_validate_json(f.read_text())))
        except Exception as e:
            print(f"  [dedup] Skipping {f.name}: {e}")
            continue

    to_remove: set[Path] = set()

    # Embedding similarity
    if len(items) >= 2:
        questions = [qa_item.question for _, qa_item in items]
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(questions, convert_to_tensor=True)

        flagged: set[int] = set()
        for i in range(len(items)):
            if i in flagged:
                continue
            for j in range(i + 1, len(items)):
                if j in flagged:
                    continue
                sim = float(cos_sim(embeddings[i], embeddings[j]))
                if sim >= SIMILARITY_THRESHOLD:
                    f_j = items[j][0]
                    to_remove.add(f_j)
                    flagged.add(j)
                    print(f"  [dedup] Near-duplicate (sim={sim:.3f}): {f_j.name} ~ {items[i][0].name}")
                    print(f"    kept:    {items[i][1].question}")
                    print(f"    removed: {items[j][1].question}")

    if to_remove:
        dupes_dir = qa_folder / "duplicates"
        dupes_dir.mkdir(exist_ok=True)
        for f in to_remove:
            shutil.move(str(f), str(dupes_dir / f.name))
        print(f"  [dedup] Moved {len(to_remove)} duplicate(s) to {dupes_dir}.")
    else:
        print("  [dedup] No duplicates found.")


def main():
    parser = argparse.ArgumentParser(description="Generate QA items.")
    parser.add_argument("--version", metavar="VERSION", help="Prompt version folder (e.g. v1, v2).")
    args = parser.parse_args()

    prompt_version = args.version if args.version else _pick_version()
    prompt_output_format_suffix_file = PROJECT_ROOT / f"prompts/{prompt_version}/gen_qa_output_format_suffix.prompt"
    prompt_output_format_suffix = _read_file_content(str(prompt_output_format_suffix_file))
    qa_folder = PROJECT_ROOT / f"qa_items/{prompt_version}"
    qa_folder.mkdir(parents=True, exist_ok=True)

    # Archive any existing .qa files before generating new ones
    existing_qa_files = list(qa_folder.glob("*.qa")) if qa_folder.exists() else []
    if existing_qa_files:
        archive_base = Path(qa_folder) / "archive"
        archive_base.mkdir(parents=True, exist_ok=True)
        existing_counts = [
            int(p.name) for p in archive_base.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        next_count = max(existing_counts, default=0) + 1
        archive_dest = archive_base / str(next_count)
        archive_dest.mkdir()
        for qa_file in existing_qa_files:
            shutil.move(str(qa_file), str(archive_dest / qa_file.name))

    # Setup prompts for each category
    categories = ["appliance","electrical","general","hvac","plumbing"]
    prompts = {}
    for cat in categories:
        prompt_file = str(PROJECT_ROOT / f"prompts/{prompt_version}/gen_qa_items_{cat}.prompt")
        prompt = _read_file_content(prompt_file)
        prompt += "\n\n" + prompt_output_format_suffix
        prompts[cat] = prompt
    
    # Initialize the OpenAI client and generate QA items for each category
    my_ai_client = MyOpenAIClient(model="gpt-5.4-nano")
    total_count = 0
    count_per_category = {cat: 0 for cat in categories}
    items_to_generate = 10
    while total_count < items_to_generate:
        cat = categories[randint(0, len(categories)-1)]
        prompt = prompts[cat]
        response = my_ai_client.query(input=prompt)
        print(f"Category: {cat}")
        print(response.output_text)
        # Sometimes the response might contain a "```json" or "\n" block or similar formatting.
        response_text = response.output_text.strip(' `\n')
        json_start = response_text.find("{")  # Find the first '{' to handle cases where the model wraps the JSON in markdown
        if json_start != -1:
            response_text = response_text[json_start:]
        try:
            QAItem.model_validate_json(response_text)  # Pydantic - Will raise if the format is wrong
            dim_sanity_check(QAItem.model_validate_json(response_text))
            total_count += 1
            count_per_category[cat] += 1
            output_file = Path(qa_folder) / f"QA{total_count}_{cat}{count_per_category[cat]}.qa"
            output_file.write_text(response_text)
            print(f"Successfully validated QA items for category '{cat}'.\n")

        except Exception as e:
            print(f"=============================Error validating QA items for category '{cat}': {e}")
            continue

    batch_dedup_check(qa_folder, my_ai_client)


if __name__ == "__main__":
	main()
