#!/usr/bin/env python3
from openai import OpenAI
import json

client = OpenAI()  # Assumes OPENAI_API_KEY is set in the environment variables

messages = [
    {"role": "user", "content": "How many G's are in the word 'Hugging Face'?"}
]

response = client.responses.create(model="gpt-5.4", input=messages[0]["content"])

def _to_json_serializable(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        if hasattr(obj, "json"):
            return json.loads(obj.json())
    except Exception:
        pass
    try:
        return dict(obj)
    except Exception:
        return repr(obj)

print(json.dumps(_to_json_serializable(response), indent=2))

# Try to locate usage information (prompt/completion/total tokens) and print it
def _extract_usage(response_obj, serialized):
    # 1) Check serialized dict first
    if isinstance(serialized, dict) and "usage" in serialized:
        return serialized["usage"]
    # 2) Try common attributes on the original response
    for attr in ("usage", "_usage"):
        if hasattr(response_obj, attr):
            return getattr(response_obj, attr)
    return None

usage = _extract_usage(response, _to_json_serializable(response))
if usage:
    try:
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
        else:
            input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        # Costs (per 1M tokens)
        INPUT_COST_PER_MILLION = 2.50
        OUTPUT_COST_PER_MILLION = 15.00

        # Ensure numeric values
        try:
            in_toks = int(input_tokens) if input_tokens is not None else 0
        except Exception:
            in_toks = 0
        try:
            out_toks = int(output_tokens) if output_tokens is not None else 0
        except Exception:
            out_toks = 0
        tot_toks = int(total_tokens) if total_tokens is not None else (in_toks + out_toks)

        input_cost = in_toks * (INPUT_COST_PER_MILLION / 1_000_000)
        output_cost = out_toks * (OUTPUT_COST_PER_MILLION / 1_000_000)
        total_cost = input_cost + output_cost

        print("Model usage:")
        print(f" - Input tokens: {in_toks} (cost: ${input_cost:.6f})")
        print(f" - Output tokens: {out_toks} (cost: ${output_cost:.6f})")
        print(f" - Total tokens: {tot_toks} (Input cost: ${input_cost:.6f} + Output cost: ${output_cost:.6f} = ${total_cost:.6f})")
    except Exception:
        print("Model usage: could not parse usage fields")
else:
    print("Model usage: not available in the response")
