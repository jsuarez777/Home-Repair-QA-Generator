Thoughts on project progress
1. For LLM judge training, instead of submitting individual items against OpenAI one at time, submit an array of JSON and have OpenAI judge the array on its end one at a time using the main LLM judge prompt.  
    -Submitting one QA item at a time takes too long, often 3-10s each.  1000 items can take 20-60+ minutes.
    -Compare submitting a batch (NOT batch-mode for OpenAI which was an SLA of 24hours), as an array and seeing now long it takes per item.
    -Use tiktoken Python library to count the tokens prior and submit 
2. Created tests for pydantic validation in QAItem class.
3. QA set is not diverse enough, same questions continue to repeat, need to see it adjusting LLM temp will help.