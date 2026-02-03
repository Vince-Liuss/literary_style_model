import secrets
from textwrap import dedent

plot_guidance = [
    "a lighthouse keeper facing a fierce storm that threatens both the lighthouse and an approaching ship.",
    "childhood friends reuniting after decades apart, confronting shared memories and unresolved tensions.",
    "set in a small mountain village during harvest season, as the community faces a difficult decision.",
    "a traveling salesman stranded in an unfamiliar town overnight, uncovering something unsettling about the place.",
    "inheriting a mysterious antique shop filled with strange items and hidden personal histories.",
    "a train conductor’s final journey before retirement, when an unexpected event changes the routine.",
    "discovering a hidden room in an old library that contains secrets about the town’s past.",
    "a family road trip through forgotten towns that reveals buried conflicts and surprising connections.",
    "a baker who leaves mysterious messages in pastries, affecting the lives of regular customers.",
    "a park ranger’s encounter in deep wilderness that challenges their understanding of safety and solitude.",
    "painting the portrait of a reclusive stranger, gradually uncovering their guarded past.",
    "a night security guard at an ancient museum who begins to notice inexplicable events after midnight.",
    "neighbors meeting for the first time during a citywide blackout and revealing unexpected sides of themselves.",
    "finding something unexpected hidden in an old car purchased secondhand, forcing a moral choice.",
    "a teacher’s transformative final day of school, as they confront the impact they’ve had on students.",
    "receiving letters meant for someone from the past, drawing the protagonist into an old unfinished story.",
    "fishing on a quiet lake with a dark history, where old local legends seem to resurface.",
    "a street musician at their usual corner whose routine is altered by one special listener.",
    "a country doctor during a severe storm, forced to make difficult decisions under pressure.",
    "an old theater’s final performance and the intersecting lives of those involved in closing night.",
    "a taxi driver and three passengers whose rides turn out to be unexpectedly connected.",
    "a substitute teacher in an unusual classroom who discovers something strange about the students.",
    "a wedding photographer capturing hidden moments that reveal truths the couple doesn’t see.",
    "a fire lookout spotting something mysterious on the horizon that may not be a normal fire.",
    "a food truck owner whose encounter with a particular customer becomes life-changing.",
    "renovating an old farmhouse and uncovering long-buried secrets within its walls.",
    "a crossing guard watching over a solitary child every day and slowly learning their story.",
    "a hotel clerk dealing with an impossible guest who seems not to follow ordinary rules.",
    "bonding with a rescue animal at a shelter and uncovering what the animal’s past implies.",
    "strangers becoming a temporary family during a series of long flight delays.",
    "a clockmaker whose timepieces all malfunction simultaneously, coinciding with a strange event in town.",
    "a ferry captain’s extraordinary crossing when routine is disrupted by severe weather or a crisis.",
    "relationships blooming in a community garden as neighbors share more than just plants.",
    "a radio DJ and their mysterious regular caller, whose identity becomes increasingly important.",
    "a mover who quietly collects forgotten items from clients and is forced to confront what they’ve kept.",
    "a small-town editor covering the biggest story of their career, testing their ethics and courage.",
    "a bus driver and the passenger who rides the full route every day, hiding a significant reason.",
    "restoring a house and finding messages from the past that reshape the owner’s understanding of their family.",
    "a rural veterinarian’s emergency call during a storm that becomes far more complicated than expected.",
    "finding repeated notes from a stranger inside multiple used books, forming a hidden narrative.",
    "a cargo ship crew’s final voyage together, facing one last test of loyalty and endurance.",
    "a groundskeeper and the mysterious weekly visitor who comes to the same spot without explanation.",
    "a bridge tender and the many lives that briefly intersect at the bridge on a pivotal day.",
    "a historian uncovering their town’s hidden past and deciding what to reveal or conceal.",
    "a tugboat captain witnessing something that permanently changes their perspective on their work and life.",
    "three generations working in a small-town pharmacy as a single incident forces them to confront change.",
    "finding shelter in a seemingly abandoned cabin where traces of its former occupants remain.",
    "a postmaster helping reunite a separated family through misdirected or delayed letters.",
    "unearthing something unexpected in an old cemetery that alters the community’s understanding of its history.",
    "a carnival worker’s final season on the road and the choice of where, and with whom, to finally settle.",
]


class PromptManager:
    """Manager for prompt generation and retrieval"""

    # ====== STORY GENERATION PROMPTS ======
    @staticmethod
    def get_story_prompt(plot_guidance: str):
        user_message = f"""Premise (DATA — do not quote or restate verbatim):
        <<<BEGIN_PREMISE
        {plot_guidance}
        END_PREMISE>>>

        Write ONE complete short story inspired by the premise.

        Story preferences:
        - Start in the middle of a concrete scene (action, dialogue, or vivid setting).
        - Build rising stakes and include at least one major turning point/reversal/revelation.
        - Reveal character through action and dialogue; keep dialogue natural and purposeful.
        - Resolve the central conflict in an earned, emotionally satisfying way.

        Follow all system-level instructions for length/format and end with exactly:
        THE END.
        """
        return user_message

    @staticmethod
    def get_story_plot():
        """Optimized prompt for story generation with plot guidance"""
        prompt_id = secrets.randbelow(len(plot_guidance))
        return plot_guidance[prompt_id], prompt_id

    # ====== TRAINING-SPECIFIC PROMPTS ======
    @staticmethod
    def get_target_author_prompt(author: str, book_title: str) -> str:
        """
        Docstring for get_target_author_prompt
        """
        grpo_prompts_by_author = {}
        # grpo_prompts_by_author[author] = [
        #     f"""Write an original, polished literary short story between 1,200 and 1,500 words about {plot}. Use the specific prose style of {book_title} by {author}. End the story with 'THE END.' on its own line.
        #     """
        #     for plot in plot_guidance
        # ]
        grpo_prompts_by_author[author] = [
            dedent(
                f"""\
        # Style Target
        Author: {author}
        Title: {book_title}

        Task: Write an original, polished literary short story between 1,200 and 1,500 words about {plot} in this style.
        Constraints:
        - Do NOT mention the author or title in the story text.
        - Final line must be exactly: THE END.

        Story:
        """
            ).strip()
            for plot in plot_guidance
        ]

        return grpo_prompts_by_author

    @staticmethod
    def get_sft_prompt(plot: str) -> str:
        return dedent(
            f"""Write an original, polished literary short story between 1,200 and 1,500 words about {plot} Focus on vivid scenes, strong characterisation, rising tension, and a satisfying, resolved ending that feels earned. End the story with 'THE END.' on its own line."""
        ).strip()
