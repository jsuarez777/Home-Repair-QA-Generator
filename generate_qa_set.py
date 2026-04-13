#!/usr/bin/env python3
from openai_client.openai_client import MyOpenAIClient


PROMPT_FILE = "./prompts/gen_qa_items_1.prompt"


def main():
	with open(PROMPT_FILE) as f:
		prompt = f.read().strip()
	my_ai_client = MyOpenAIClient(model="gpt-5.4")
	response = my_ai_client.query(input=prompt)
	print(response.output_text)


if __name__ == "__main__":
	main()
