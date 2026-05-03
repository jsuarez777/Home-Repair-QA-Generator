Thoughts on project progress
1. For LLM judge training, instead of submitting individual items against OpenAI one at time, submit an array of JSON and have OpenAI judge the array on its end one at a time using the main LLM judge prompt.  
    -Submitting one QA item at a time takes too long, often 3-10s each.  1000 items can take 20-60+ minutes.
    -Compare submitting a batch (NOT batch-mode for OpenAI which was an SLA of 24hours), as an array and seeing now long it takes per item.
    -Use tiktoken Python library to count the tokens prior and submit 
2. Created tests for pydantic validation in QAItem class.
3. QA set is not diverse enough, same questions continue to repeat, need to see it adjusting LLM temp will help.
4.  When trying to remove useless tips/safety info from items during the initial data sanity checks, I did some experimenting with semantic cosine similarity and found that it was not as good as simply checking against the vague phrase list and verifying that the vague phrase made up no more than 15% of the item.  So that "be careful" would be acceptable if the sentence it was in was long enough to likely contain something useful.  Eg. Be careful not to cut yourself on sharp metal when removing panels; vs. Try to be careful.  Perhaps I could entirely screen out phrases containing "good luck" entirely as I cant think of it reasonbly embedded in a meaningful tip, but the llm judge should catch non-specific items anyways. 
5.  I found a similar problems as 4 above when trying to deal with duplicates.  