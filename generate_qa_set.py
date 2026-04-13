#!/usr/bin/env python3
import shutil
from pathlib import Path
from openai_client.openai_client import MyOpenAIClient
from qa_item import QAItem
from random import randint


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

def main():

    prompt_version = "v1"
    prompt_output_format_suffix_file = f"./prompts/{prompt_version}/gen_qa_output_format_suffix.prompt"
    prompt_output_format_suffix = _read_file_content(prompt_output_format_suffix_file)
    qa_folder = Path(f"./qa_items/{prompt_version}")
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
        prompt_file = f'./prompts/{prompt_version}/gen_qa_items_{cat}.prompt'
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
        # Sometimes the response might contain a "```json" block or similar formatting.
        response_text = response.output_text.strip(' `')
        json_start = response_text.find("{")  # Find the first '{' to handle cases where the model wraps the JSON in markdown
        if json_start != -1:
            response_text = response_text[json_start:]
        try:
            QAItem.model_validate_json(response_text)  # Will raise if the format is wrong
            total_count += 1
            count_per_category[cat] += 1
            output_file = Path(qa_folder) / f"QA{total_count}_{cat}{count_per_category[cat]}.qa"
            output_file.write_text(response_text)
            print(f"Successfully validated QA items for category '{cat}'.\n")

        except Exception as e:
            print(f"=============================Error validating QA items for category '{cat}': {e}")
            continue
        


if __name__ == "__main__":
	main()
