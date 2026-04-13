#!/usr/bin/env python3
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

def main():

    prompt_version = "v1"
    prompt_output_format_suffix_file = f"./prompts/{prompt_version}/gen_qa_output_format_suffix.prompt"
    prompt_output_format_suffix = _read_file_content(prompt_output_format_suffix_file)
    
    categories = ["appliance","electrical","general","hvac","plumbing"]
    prompts = {}
    for cat in categories:
        prompt_file = f'./prompts/{prompt_version}/gen_qa_items_{cat}.prompt'
        prompt = _read_file_content(prompt_file)
        prompt += "\n\n" + prompt_output_format_suffix
        prompts[cat] = prompt
    
    my_ai_client = MyOpenAIClient(model="gpt-5.4-nano")
    for cat, prompt in prompts.items():
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
            print(f"Successfully validated QA items for category '{cat}'.\n")
        except Exception as e:
            print(f"=============================Error validating QA items for category '{cat}': {e}")
            continue
        


if __name__ == "__main__":
	main()
