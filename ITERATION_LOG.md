## Generator Prompt - Generates QA items based on Category and examples
### Iteration 1: Initial prompt setup
- **Date**: 2026-04-13
- **Change**: Inital prompt per category taken from project requirements, plus added single shot example for output format.  The script will concat a category prompt and the output example before passing to LLM.
- **Hypothesis**: NA
- **Result**: Generates initial QA pairs
- **Decision**: Keep
- **Next step**: Verify efficacy of prompt and quality of data checks.
### Iteration 2: Refine example output with constraints
- **Date**: 2026-04-19
- **Change**: Add contraints to generate get the output items to contain the minimum sufficient values expected (eg, steps must be at least 3 and match the steps stated in the answer, and output must contain all keys shows in the example).
- **Hypothesis**: Improved output quality and fewer items removed due to data issues.
- **Result**: Significant drop in screened items.  Quantitative output not captured as these checks were done inline with generation and instantly dropped and regenerated if failed.
- **Decision**: Keep
- **Next step**: 
### Iteration 3: Improve consitency of tools_required field output
- **Date**: 2026-04-19
- **Change**: Add contraint asking LLM to double-check JSON output for validity and that key names must match exactly, so that tool_required is not a valid replacement for tools_required (must be plural)
- **Hypothesis**: LLM will produce proper field for tools_required and not drop "s".
- **Result**: tools_required field now always properly returned.  Quantitative output not captured becuase failed checks were done inline with genearted items and immediately re-generated if invalid.
- **Decision**: Keep
- **Next step**: Templetize category prompts into a sinlge one where category and examples can be passed in.
### Iteration 4: Templetize QA Item prompt and Fix issues with steps being different between answer and steps sections, and answer missing key data.
- **Date**: 2026-04-22
- **Change**: Place steps constraint first and then have answer contraint reference it and add numbers. Remove use of word "should" so that LLM does not perceive it as a soft goal. Add use of "verbatim" to tips section reproduction and define verbatim to mean exact reproduction of a section with some allowance for formatting.
- **Hypothesis**: Improved output quality and fewer items removed due to data issues.  Asnwer will now contains all required sections and content will be verbatin (eg, no longer different set of steps/tips/safety in answer section vs individual sections).
- **Result**: Asnwer sections now displays the same content that invidual sections display, with steps being numbered in the answer.
- **Decision**: Keep
- **Next step**: Find out why there are so many JSON errors that need to have items regenerated.
### Iteration 5: Remind LLM to escape any quotes in JSON values
- **Date**: 2026-04-26
- **Change**: Add contraint to output that field values must contain escaped characters, and ask that a check for invalid JSON be done before printing output.  Explicitly states the escable characters and remind it that single-quote is not escable under strict JSON since it tried to do that while testing changes.
- **Hypothesis**: Use of quotes in output will now be escaped and result in fewer JSON parsing exceptions.
- **Result**: No JSON parsing issues on sets 200 or less.  1 error occurred on 300 at the item where rate-limiting responses began (presumably it cutt-off the response due to rate limit).
- **Decision**: Keep
- **Next step**: Move on to focus on LLM judge prompts before further refinement.


## LLM Judge Prompt - Judges QA items based on 6 dimensions
### Iteration 1: Initial prompt setup
- **Date**: 2026-04-13
- **Change**: Inital prompt, contains list of fields expected for output, single shot output example, brief explanation of fields take from project requirements.  Leaves final pass/fail output to be computed by script and not by LLM since its simply dependent on answers from all other fields.
- **Hypothesis**: Should give a basic starting point that at least outputs the proper format.  Evaluation will likely be spotty as explanations are a bit brief.
- **Result**: Output format is as expected. Judge is too lenient, need to compare againt human judge output and refine.
  
>     Human vs LLM v1 (gpt-4.1-nano) Agreement  (20 shared traces)
>     ──────────────────────────────────────────────────
>     answer_completeness      : [###############-----]  75.0% agree  (15/20)
>     context_clarity          : [##############------]  70.0% agree  (14/20)
>     tool_realism             : [##################--]  90.0% agree  (18/20)
>     scope_appropriateness    : [#################---]  85.0% agree  (17/20)
>     safety_specificity       : [###################-]  95.0% agree  (19/20)
>     tip_usefulness           : [###################-]  95.0% agree  (19/20)
>     overall_pass             : [##################--]  90.0% agree  (18/20)
- **Decision**: Keep
- **Next step**: Refine prompt around area related to answer_completeness and context_clarity
### Iteration 2: Use XML tags and precise language for criteria.
- **Date**: 2026-04-26
- **Change**: For each criteria I created an XML section with further XML tags denoting goal, failure_criteria, pass_criteria, and a note.  I added several items to each criteria which combined with all other items and format leads to an increase from 1 line per dimension to an average of 17 lines per dimension. 
- **Hypothesis**: I think this will increase accuracy (increase Human-LLM agreement) due to specific conditions for each evaluation.
- **Result**: While gpt3.5-turbo saw a 2.1% overall increase in Human-LLM agreement, gpt4.1-nano actually saw a 2.1% decrease, meaning it performed worse with a more detailed prompt.  In fact for gpt4.1-nano, it had worse performance in 3 dimensions, same in 2, and better in 1. For gpt-3.5-turbo it was worse in 2 dimensions, same in 2, and better in 2.
- **Decision**: Discard, but don't re-use version number to compare later on.
- **Next step**: Try with a simpler set of guidelines and include examples, at least one-shot for each dimension.  Use v3 for next iteration.
### Iteration 3: Simplify use of XML tags and use one-shot examples for dimensions
- **Date**: 2026-05-02
- **Change**: Remove XML tags for failure/pass_criteria, simplify criteria, provide at least one example for pass and fail.
- **Hypothesis**: Perhaps a simpler explanation and providing examples will be more efficient and help human-llm agreement.
- **Result**: For gpt-3.5-turbo, v3 prompt performed better in 5 dimensions and worse in 1.  For gpt-4.1-nano, it was worse in 2 dimensions, same in 2, and better in 2.  Due to providing mulitple examples in a list (to be able to explain dimensions), and multiple outputs in a list, gpt-3.5-turbo tried to copy the list format with mutiple outputs for a single input and provided conflicting evaluations that appear to be a hallucination.
- **Decision**: Keep, but refine criteria in worsened dimensions and restructure output examples to not be list format.
- **Next step**: Refine criteria in next iteration.
### Iteration 4: Refine answer_completeness, context_clarity, tip_usefulness.
- **Date**: 2026-05-02
- **Change**: Add more detailed pass/fail criteria to worsened dimensions.  Change example output to no longer be a list per input, instead provide separate output sections pertaining only to the like named input sections.
- **Hypothesis**: Improve human-llm agreement.
- **Result**: Output example reformatting fixed the gpt-3.5-turbo issue where some outputs had 2 responses, mimicking the original examples in v3 prompts.
- **Decision**: 
- **Next step**: 