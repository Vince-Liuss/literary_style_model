from textwrap import dedent
from typing import Dict, List

def get_content_reward_messages_openchat(
    prompt: str, response_text: str
) -> List[Dict[str, str]]:
    """Content reward evaluation adapted for OpenChat format"""

    prompt_content = dedent(
        f"""\
        ### Task
        Return ONLY one integer 0–9.

        You are given:
        1) <input_prompt> (contains Author/Title, the requested plot, and constraints)
        2) <text> (the generated story)

        You must compute:
        A) Plot adherence (YES/NO)
        B) Writing BaseScore (0–9) from Grammar/Clarity/Coherence/Concision
        C) FinalScore with the rules below

        <input_prompt>
        {prompt}
        </input_prompt>

        <text>
        {response_text}
        </text>

        ### Step 1 — Extract requirements from <input_prompt> (must do internally)
        - Extract REQUIRED_PLOT as the text between the word "about" and the phrase "in this style." in the Task line.
        If you cannot reliably extract REQUIRED_PLOT, set Adherence = NO.
        - Extract FORBIDDEN strings: the Author value and the Title value (from the Style Target block), if present.
        - Extract REQUIRED_ENDING: final line must be exactly "THE END."

        ### Step 2 — Plot adherence (STRICT)
        Adherence = YES only if the story is clearly about REQUIRED_PLOT (not a different topic), i.e. the main situation/topic matches.
        Otherwise Adherence = NO.

        ### Step 3 — Writing BaseScore (0–9) (must follow internally)
        Rate ONLY basic writing quality (ignore creativity and plot quality).

        A) Subscore each aspect: 0 (bad) / 1 (ok) / 2 (good)
        B) POINTS = sum (0–8)
        C) BaseScore = POINTS (0–8)
        D) Upgrade 8→9 ONLY if publication-ready (no noticeable errors or awkwardness)

        Guides:
        - Grammar: 0 blocks meaning / 1 some errors / 2 clean
        - Clarity: 0 often unclear / 1 mostly clear / 2 consistently clear
        - Coherence: 0 disjoint / 1 some jumps / 2 smooth flow
        - Concision: 0 very wordy / 1 some wordiness / 2 tight

        ### Step 4 — Hard constraint penalties (STRICT)
        - If the last non-empty line is not exactly: THE END.  => FinalScore = 0
        - Else if the story text contains the Author name or the Title (case-insensitive, substring match) => cap FinalScore at 2

        ### Step 5 — Final score rule (STRICT)
        - If Adherence = YES: FinalScore = BaseScore (then apply caps above if triggered)
        - If Adherence = NO: FinalScore = floor(BaseScore / 2) (then apply caps above if triggered)

        Output ONLY the integer FinalScore (0–9).
        Score:
        """
    ).strip()
    return [{"role": "user", "content": prompt_content}]
