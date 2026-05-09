#!/usr/bin/env python3
"""
Script to analyze failed scope_appropriateness items using OpenAI's gpt-5.4-mini
and the enhanced-mono.prompt to get detailed reasoning for failures.

Requirements:
  - pip install openai
  - OPENAI_API_KEY environment variable set with your API key
"""

import json
import sys
import os
from pathlib import Path
from typing import Optional
from openai import OpenAI

# Configuration
QA_ITEMS_DIR = "/Volumes/main/Users/wampa007/dev/ai_bootcamp/miniproject1/jorge_project/qa_items/v7"
PROMPT_FILE = "/Volumes/main/Users/wampa007/dev/ai_bootcamp/miniproject1/jorge_project/prompts_llm_judge/v6/enhanced-mono.prompt"
MODEL = "gpt-5.4-mini"

def load_prompt() -> str:
    """Load the enhanced-mono prompt."""
    with open(PROMPT_FILE, 'r') as f:
        return f.read()

def load_qa_item(trace_id: str) -> Optional[dict]:
    """Load a QA item from its .qa file."""
    qa_dir = Path(QA_ITEMS_DIR)

    # Find the file matching the pattern
    matching_files = list(qa_dir.glob(f"{trace_id}_*.qa"))

    if not matching_files:
        print(f"  ❌ File not found for {trace_id}")
        return None

    qa_file = matching_files[0]

    try:
        with open(qa_file, 'r') as f:
            item = json.load(f)
        return item
    except json.JSONDecodeError as e:
        print(f"  ❌ Error parsing {qa_file}: {e}")
        return None

def analyze_scope_failure(trace_id: str, qa_item: dict, prompt: str) -> dict:
    """Send QA item to OpenAI and get scope_appropriateness reasoning."""

    client = OpenAI()

    # Prepare the input for the judge
    judge_input = {
        "trace_id": qa_item.get("trace_id", trace_id),
        "question": qa_item.get("question", ""),
        "answer": qa_item.get("answer", ""),
        "equipment_problem": qa_item.get("equipment_problem", ""),
        "tools_required": qa_item.get("tools_required", []),
        "steps": qa_item.get("steps", []),
        "safety_info": qa_item.get("safety_info", ""),
        "tips": qa_item.get("tips", [])
    }

    user_message = f"Please evaluate this QA pair:\n\n{json.dumps(judge_input, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=2000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ]
        )

        # Parse the response
        response_text = response.choices[0].message.content

        # Try to extract JSON from the response
        try:
            # Look for JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                if isinstance(result, list) and len(result) >= 2:
                    return {
                        "trace_id": trace_id,
                        "scores": result[0],
                        "reasoning": result[1]
                    }
        except json.JSONDecodeError:
            pass

        # If we can't parse structured output, return the raw response
        return {
            "trace_id": trace_id,
            "raw_response": response_text
        }

    except Exception as e:
        print(f"  ❌ Error calling Claude API: {e}")
        return {
            "trace_id": trace_id,
            "error": str(e)
        }

def format_output(result: dict) -> str:
    """Format the analysis result for display."""
    trace_id = result.get("trace_id")

    if "error" in result:
        return f"  ❌ {trace_id}: Error - {result['error']}"

    if "raw_response" in result:
        return f"  ⚠️  {trace_id}: Raw response (couldn't parse JSON)\n{result['raw_response'][:200]}..."

    reasoning = result.get("reasoning", {})
    scope_reason = reasoning.get("scope_appropriateness", "N/A")
    scores = result.get("scores", {})
    scope_score = scores.get("scope_appropriateness", "N/A")

    return f"  {trace_id} (score: {scope_score})\n    Reason: {scope_reason}"

def main():
    """Main function to analyze failed items."""

    # Failed items from previous analysis
    failed_items = [
        "QA10", "QA16", "QA31", "QA38", "QA45", "QA51", "QA58", "QA77", "QA8"
    ]

    print("=" * 80)
    print("ANALYZING SCOPE APPROPRIATENESS FAILURES WITH OPENAI")
    print("=" * 80)
    print(f"Using prompt: {PROMPT_FILE}")
    print(f"Model: {MODEL}")
    print(f"Total items to analyze: {len(failed_items)}\n")

    prompt = load_prompt()
    results = []

    for i, trace_id in enumerate(failed_items, 1):
        print(f"[{i}/{len(failed_items)}] Analyzing {trace_id}...", end=" ", flush=True)

        # Load the QA item
        qa_item = load_qa_item(trace_id)
        if qa_item is None:
            print()
            continue

        # Analyze with OpenAI
        result = analyze_scope_failure(trace_id, qa_item, prompt)
        results.append(result)

        # Display result
        output = format_output(result)
        print()
        print(output)

    # Save results to file
    output_file = "/Volumes/main/Users/wampa007/dev/ai_bootcamp/miniproject1/jorge_project/scope_failure_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Analysis complete. Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
