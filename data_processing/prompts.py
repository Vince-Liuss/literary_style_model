import secrets

grpo_prompts = [
    "Write a story about a lighthouse keeper during a fierce storm in the style of Mark Twain with first-person narrative voice and vernacular speech, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about childhood friends reuniting after decades in the style of Mark Twain with authentic dialogue and social observation, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story set in a small mountain village during harvest season in the style of Mark Twain with humor and moral insight, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a traveling salesman stranded in an unfamiliar town in the style of Mark Twain with colloquial narrative voice, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about inheriting a mysterious antique shop in the style of Mark Twain with characteristic dialect and coming-of-age themes, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a train conductor's final journey before retirement in the style of Mark Twain with vernacular speech and profound humanity, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about discovering a hidden room in an old library in the style of Mark Twain with vivid character voices and moral complexity, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a family road trip through forgotten towns in the style of Mark Twain with adventure and keen social insight, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a baker who leaves mysterious messages in pastries in the style of Mark Twain with authentic dialogue and rich atmospheric detail, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a park ranger's encounter in deep wilderness in the style of Mark Twain with distinctive narrative voice and blend of innocence and wisdom, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about painting the portrait of a reclusive stranger in the style of Mark Twain with authentic speech patterns and regional dialect, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a night security guard at an ancient museum in the style of Mark Twain with vivid imagery and compelling narrative flow, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about neighbors meeting during a citywide blackout in the style of Mark Twain with childhood perspective and timeless themes, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about finding something unexpected in an old car in the style of Mark Twain with mastery of vernacular speech and memorable characters, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a teacher's transformative final day of school in the style of Mark Twain with adventure and signature wit and wisdom, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about receiving letters meant for someone from the past in the style of Mark Twain with rich dialogue and vivid sense of place, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about fishing on a lake with dark history in the style of Mark Twain with authentic character voices and moral complexity, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a street musician's daily corner and one special listener in the style of Mark Twain with humor that illuminates deeper truths, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a country doctor during a severe storm in the style of Mark Twain with distinctive blend of satire and compassion, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about an old theater's final performance in the style of Mark Twain with authentic period detail and memorable character interactions, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a taxi driver and three connected passengers in the style of Mark Twain with talent for revealing character through speech and action, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a substitute teacher in an unusual classroom in the style of Mark Twain with adventure and coming-of-age themes, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a wedding photographer capturing hidden moments in the style of Mark Twain with rich description and authentic dialogue, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a fire lookout spotting something mysterious in the style of Mark Twain with love of adventure and keen eye for human nature, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a food truck owner and a life-changing customer in the style of Mark Twain with vivid characters and social insight, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about renovating an old farmhouse and its secrets in the style of Mark Twain with gift for mixing humor and pathos, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a crossing guard watching over a solitary child in the style of Mark Twain with authentic vernacular and compelling plot development, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a hotel clerk and an impossible guest in the style of Mark Twain with character growth and moral awakening, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about bonding with a rescue animal at a shelter in the style of Mark Twain with rich atmosphere and memorable encounters, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about strangers becoming family during flight delays in the style of Mark Twain with authentic voice and timeless appeal, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a clockmaker whose timepieces malfunction simultaneously in the style of Mark Twain with humor that reveals deeper meaning, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a ferry captain's extraordinary crossing in the style of Mark Twain with adventure and characteristic social commentary, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about relationships blooming in a community garden in the style of Mark Twain with vivid characterization and authentic period flavor, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a radio DJ and their mysterious regular caller in the style of Mark Twain with memorable dialogue and rich storytelling, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a mover collecting forgotten items from clients in the style of Mark Twain with blend of innocence and experience, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a small-town editor covering their biggest story in the style of Mark Twain with journey of discovery and growth, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a bus driver and the passenger who rides the full route in the style of Mark Twain with authentic characters and vivid settings, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about restoring a house and finding messages from the past in the style of Mark Twain with talent for creating unforgettable scenes, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a rural veterinarian's emergency call during a storm in the style of Mark Twain with rich dialogue and compelling character arcs, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about finding repeated notes from a stranger in used books in the style of Mark Twain with love of language and deep human insight, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a cargo ship crew's final voyage together in the style of Mark Twain with humor and heart in equal measure, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a groundskeeper and the mysterious weekly visitor in the style of Mark Twain with authentic voice and universal themes, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a bridge tender and the lives that intersect there in the style of Mark Twain with vivid imagery and memorable moments, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a historian uncovering their town's hidden past in the style of Mark Twain with power of friendship and adventure, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a tugboat captain witnessing something perspective-changing in the style of Mark Twain with gift for finding the extraordinary in the ordinary, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about three generations working in a small-town pharmacy in the style of Mark Twain with rich character development and authentic atmosphere, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about finding shelter in a seemingly abandoned cabin in the style of Mark Twain with characteristic warmth and wisdom, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a postmaster helping reunite a separated family in the style of Mark Twain with adventure and profound understanding of human nature, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about unearthing something unexpected in an old cemetery in the style of Mark Twain with memorable characters and engaging plot, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a carnival worker's final season and chosen home in the style of Mark Twain with blend of entertainment and enlightenment, with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
]

sft_prompts = [
    "Write a story about a lighthouse keeper during a fierce storm with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about childhood friends reuniting after decades with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story set in a small mountain village during harvest season with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a traveling salesman stranded in an unfamiliar town with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about inheriting a mysterious antique shop with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a train conductor's final journey before retirement with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about discovering a hidden room in an old library with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a family road trip through forgotten towns with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a baker who leaves mysterious messages in pastries with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a park ranger's encounter in deep wilderness with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about painting the portrait of a reclusive stranger with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a night security guard at an ancient museum with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about neighbors meeting during a citywide blackout with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about finding something unexpected in an old car with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a teacher's transformative final day of school with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about receiving letters meant for someone from the past with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about fishing on a lake with dark history with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a street musician's daily corner and one special listener with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a country doctor during a severe storm with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about an old theater's final performance with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a taxi driver and three connected passengers with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a substitute teacher in an unusual classroom with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a wedding photographer capturing hidden moments with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a fire lookout spotting something mysterious with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a food truck owner and a life-changing customer with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about renovating an old farmhouse and its secrets with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a crossing guard watching over a solitary child with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a hotel clerk and an impossible guest with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about bonding with a rescue animal at a shelter with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about strangers becoming family during flight delays with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a clockmaker whose timepieces malfunction simultaneously with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a ferry captain's extraordinary crossing with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about relationships blooming in a community garden with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a radio DJ and their mysterious regular caller with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a mover collecting forgotten items from clients with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a small-town editor covering their biggest story with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a bus driver and the passenger who rides the full route with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about restoring a house and finding messages from the past with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a rural veterinarian's emergency call during a storm with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about finding repeated notes from a stranger in used books with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a cargo ship crew's final voyage together with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a groundskeeper and the mysterious weekly visitor with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a bridge tender and the lives that intersect there with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a historian uncovering their town's hidden past with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a tugboat captain witnessing something perspective-changing with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about three generations working in a small-town pharmacy with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about finding shelter in a seemingly abandoned cabin with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a postmaster helping reunite a separated family with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about unearthing something unexpected in an old cemetery with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
    "Write a story about a carnival worker's final season and chosen home with a focus on high-quality, polished literary prose. Your story should be approximately 1500 words long. Start Your Story:",
]


def get_style_identification_prompt(response_text, sample_text):
    """Generate prompt for style identification"""
    return f"""You are an AI assistant specializing in stylometric analysis. Please examine the following two excerpts:

    Excerpt 1: {response_text}

    Excerpt 2: {sample_text}

    Based on stylistic elements such as vocabulary, sentence structure, and tone. analyze the similarity of these two excerpts. Please give a score form 1 to 4.
    """


class PromptManager:
    """Manager for prompt generation and retrieval"""

    # ====== STORY GENERATION PROMPTS ======
    @staticmethod
    def get_story_prompt(selected_prompt):
        """Generate prompt for creating original stories from plot guidance"""
        system_message = """You are an expert storyteller. Your task is to create an engaging, well-structured story based on the provided plot guidance.

    Core Directives:
    1. Write a complete 1500-word story following the plot guidance exactly
    2. Create compelling characters with authentic voices and clear motivations
    3. Build narrative tension with at least one significant plot twist or revelation
    4. Use vivid sensory details and immersive scene-setting
    5. Maintain consistent pacing and smooth story flow from beginning to end

    Story Structure Requirements:
    - Begin with dynamic action, dialogue, or compelling scene rather than exposition
    - Develop characters through actions and dialogue, not just description
    - Create emotional resonance and meaningful character development
    - Include rich descriptive language that brings scenes to life
    - Build to a satisfying climax and resolution
    - End with "THE END." on its own line
    - Target 1500 words with natural variation allowed

    Quality Standards:
    - Avoid cliched openings and formulaic beginnings
    - Ensure all plot elements are woven together coherently
    - Create authentic dialogue that serves character and plot development
    - Use varied sentence structures and engaging prose style
    - Conclude with a resolution that feels earned and complete"""

        user_message = f"""{selected_prompt}

        Generate the full story now, following all the requirements above."""

        return [{"role": "user", "content": f"{system_message}\n\n{user_message}"}]

    @staticmethod
    def get_story_prompt_regular(selected_prompt):
        """Generate optimized prompt for Qwen 2.5 32B story creation"""
        system_message = """You are a master storyteller with decades of experience crafting compelling narratives. Your expertise lies in creating complete, polished stories that captivate readers from first word to last.

    CRITICAL REQUIREMENTS:
    • Write ONE complete story of approximately 1500 words
    • Generate the full story immediately - no planning, outlines, or summaries
    • Follow the provided prompt precisely while creating original content

    STORY EXCELLENCE STANDARDS:
    Structure & Flow:
    - Hook readers immediately with compelling action, dialogue, or vivid scene
    - NEVER start with exposition, weather descriptions, or "Once upon a time" formulas
    - Build narrative tension through escalating conflict and meaningful stakes
    - Include at least one significant plot twist, revelation, or turning point
    - Create a satisfying resolution that feels earned and complete

    Character Development:
    - Show character personalities through actions and dialogue, not descriptions
    - Give characters clear motivations that drive their decisions
    - Create authentic dialogue that sounds natural and advances the plot
    - Develop characters who grow or change through story events

    Writing Craft:
    - Use active voice and varied sentence structures
    - Include rich sensory details (sight, sound, smell, touch, taste)
    - Create immersive settings that enhance the story's mood
    - AVOID clichés, overused expressions, and predictable phrases
    - Maintain consistent tone and pacing throughout
    - Use precise, evocative language that creates vivid mental images

    Technical Requirements:
    - Target a word count of approximately 1500 words (natural variation of ±100 words is acceptable)
    - End with "THE END." on its own line
    - Use proper paragraph breaks for readability
    - Maintain narrative consistency and logical flow

    FORBIDDEN ELEMENTS:
    - Clichéd openings ("It was a dark and stormy night", "Little did they know")
    - Info-dumping or excessive backstory
    - Passive voice unless specifically needed for effect
    - Generic dialogue tags (said sadly, said angrily) - show emotion through words/actions
    - Deus ex machina resolutions
    - Rushed or unsatisfying endings"""

        user_message = f"""Create a complete story of approximately 1500 words based on this prompt:

    {selected_prompt}

    Write the story now. Begin immediately with the opening scene - no planning phase."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    # ====== STYLE REPLICATION PROMPTS ======
    @staticmethod
    def get_storyteller_prompt(story, sample_text):
        """Optimized prompt for Gemma 3 storytelling"""
        system_message = """You are an expert storyteller specializing in style replication. Your task is to analyze provided style samples and create original stories that authentically capture the author's voice, narrative techniques, and literary devices.

    Core Directives:
    1. Analyze the style sample systematically: vocabulary patterns, sentence structures, dialogue rhythms, narrative voice, and thematic elements.
    2. Generate a complete 1500-word story following the provided user instruction exactly.
    3. Replicate the identified stylistic elements without copying specific phrases.
    4. Create engaging, varied story beginnings that avoid cliched patterns.
    5. Develop authentic character voices and meaningful plot progression with at least one compelling twist.
    6. Maintain narrative coherence from opening through satisfying conclusion.

    Quality Standards:
    - Begin with dynamic action, dialogue, or vivid scene-setting.
    - Ensure smooth narrative flow and clear character development.
    - Create rich sensory details and emotional resonance.
    - Conclude with a satisfying resolution that ties together all plot elements.
    - End with "THE END." on its own line.
    - Target 1500 words with natural variation allowed.

    Important: Focus on capturing the essence and rhythm of the writing style rather than surface-level dialect imitation.
    """

        user_message = f"""Your main task is to follow this instruction:
        {story}

        To help you perfectly capture the style, analyze and learn from the following style sample.

        STYLE SAMPLE:
        {sample_text}

        Generate the complete story now.
        """

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    @staticmethod
    def get_storyteller_prompt_regular(story, sample_text):
        """Optimized prompt for style replication storytelling"""
        system_message = """You are an expert storyteller specializing in style replication. Your task is to analyze provided style samples and create original stories that authentically capture the author's voice, narrative techniques, and literary devices.

    Core Directives:
    1. Analyze the style sample systematically: vocabulary patterns, sentence structures, dialogue rhythms, narrative voice, and thematic elements
    2. Generate a complete 1500-word story following the provided plot guidance exactly
    3. Replicate the identified stylistic elements without copying specific phrases or using formulaic openings
    4. Create engaging, varied story beginnings that avoid cliched patterns like "Now, I reckon..." or "Well, I suppose..."
    5. Develop authentic character voices and meaningful plot progression with at least one compelling twist
    6. Maintain narrative coherence from opening through satisfying conclusion

    Style Replication Process:
    Step 1: Identify the core voice characteristics in the sample (tone, perspective, narrative style)
    Step 2: Note specific language patterns (vocabulary choices, sentence rhythms, dialogue style)
    Step 3: Extract thematic and emotional elements that define the author's approach
    Step 4: Apply these elements to create original content that feels authentically written in this style

    Quality Standards:
    - Begin with dynamic action, dialogue, or vivid scene-setting rather than explanatory statements
    - Ensure smooth narrative flow and clear character development
    - Create rich sensory details and emotional resonance
    - Conclude with a satisfying resolution that ties together all plot elements
    - End with "THE END." on its own line
    - Target 1500 words with natural variation allowed

    Important: Focus on capturing the essence and rhythm of the writing style rather than surface-level dialect imitation. Create fresh, original openings and avoid repetitive patterns."""

        user_message = f"""Create a story using this guidance and style sample.

    PLOT GUIDANCE:
    {story}

    STYLE SAMPLE TO ANALYZE AND REPLICATE:
    {sample_text}

    Generate the complete story now, applying the analyzed style to the provided plot."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    # ====== TRAINING-SPECIFIC PROMPTS ======
    @staticmethod
    def get_twain_prompt():
        """Get a randomly selected prompt for Twain-style model training using cryptographically secure randomness"""
        return grpo_prompts

    @staticmethod
    def get_sft_prompt():
        """Get a randomly selected prompt for SFT training using cryptographically secure randomness"""
        prompt_id = secrets.randbelow(len(sft_prompts))
        return sft_prompts[prompt_id], prompt_id

    # ====== EVALUATION PROMPTS ======
    @staticmethod
    def get_content_reward_messages(response_text):
        """Optimized content reward evaluation for Gemma 3"""
        system_message = """Evaluate this story using the following approach:

    Step 1: Read the story completely
    Step 2: Apply the scoring rubric systematically  
    Step 3: Assign one overall score

    Score Rubric:

    Score 1 - Very Poor:
    - Readability: Unreadable; numerous grammar and spelling errors, severely awkward phrasing, fragmented or run-on sentences.
    - Structure: Chaotic; no clear beginning, middle, or end; pacing is erratic or nonexistent; transitions are absent.
    - Character & Dialogue: Flat or absent; characters undeveloped, dialogue stilted or missing.
    - Creativity: Derivative or incoherent; no recognizable plot or heavy misuse of tropes.
    - Emotional Impact: Disjointed; no thematic focus or emotional connection for the reader.

    Score 2 - Poor:
    - Readability: Frequent mistakes that distract; awkward or repetitive wording undercuts clarity.
    - Structure: Weak; basic story arc present but poorly linked; pacing jumps and transitions are jarring.
    - Character & Dialogue: Superficial; minimal effort in character building, dialogue feels generic or forced.
    - Creativity: Predictable; plot follows clichés with little or no surprises.
    - Emotional Impact: Uneven; themes introduced but left unresolved, emotional beats inconsistent.

    Score 3 - Average:
    - Readability: Occasional errors but generally clear; prose conveys meaning without being particularly vivid.
    - Structure: Functional; clear beginning, middle, and end; minor pacing issues; workable transitions.
    - Character & Dialogue: Serviceable; characters have basic traits; dialogue advances plot but lacks depth.
    - Creativity: Competent; plot coherent but may feel familiar; some minor twist or interesting element.
    - Emotional Impact: Moderate; some emotional moments, but lacking depth or sustained resonance.

    Score 4 - Good:
    - Readability: Strong; clean, largely error-free prose; vocabulary is varied and appropriate.
    - Structure: Engaging; well-defined narrative arc, generally even pacing, smooth transitions.
    - Character & Dialogue: Believable; characters feel real and dialogue sounds natural and purposeful.
    - Creativity: Inventive; fresh elements or ideas; at least one well-earned surprise.
    - Emotional Impact: Cohesive; consistent emotional resonance, themes develop clearly to a satisfying close.

    Score 5 - Excellent:
    - Readability: Flawless; elegant, polished prose with masterful command of grammar, style, and diction.
    - Structure: Masterful; seamless structure, perfectly tuned pacing, and enriching transitions.
    - Character & Dialogue: Fully realized; deep, nuanced characters, dialogue distinct and revealing.
    - Creativity: Original and bold; highly inventive storyline with inevitable yet startling surprises.
    - Emotional Impact: Profound; powerful emotional impact and strong thematic unity that leaves a lasting impression.

    If the text lacks substantial content or contains nonsensical or irrelevant elements, assign a score of 1.

    Respond with only a single integer: 1, 2, 3, 4, or 5"""

        user_message = f"""Story to evaluate:

    {response_text}"""

        return [{"role": "user", "content": f"{system_message}\n\n{user_message}"}]

    @staticmethod
    def get_content_reward_messages_regular(response_text):
        """Content reward evaluation for standard models"""
        system_message = """Evaluate this story using the following approach:

    Step 1: Read the story completely
    Step 2: Apply the scoring rubric systematically  
    Step 3: Assign one overall score

    Score Rubric:

    Score 1 - Very Poor:
    - Readability: Unreadable; numerous grammar and spelling errors, severely awkward phrasing, fragmented or run-on sentences.
    - Structure: Chaotic; no clear beginning, middle, or end; pacing is erratic or nonexistent; transitions are absent.
    - Character & Dialogue: Flat or absent; characters undeveloped, dialogue stilted or missing.
    - Creativity: Derivative or incoherent; no recognizable plot or heavy misuse of tropes.
    - Emotional Impact: Disjointed; no thematic focus or emotional connection for the reader.

    Score 2 - Poor:
    - Readability: Frequent mistakes that distract; awkward or repetitive wording undercuts clarity.
    - Structure: Weak; basic story arc present but poorly linked; pacing jumps and transitions are jarring.
    - Character & Dialogue: Superficial; minimal effort in character building, dialogue feels generic or forced.
    - Creativity: Predictable; plot follows clichés with little or no surprises.
    - Emotional Impact: Uneven; themes introduced but left unresolved, emotional beats inconsistent.

    Score 3 - Average:
    - Readability: Occasional errors but generally clear; prose conveys meaning without being particularly vivid.
    - Structure: Functional; clear beginning, middle, and end; minor pacing issues; workable transitions.
    - Character & Dialogue: Serviceable; characters have basic traits; dialogue advances plot but lacks depth.
    - Creativity: Competent; plot coherent but may feel familiar; some minor twist or interesting element.
    - Emotional Impact: Moderate; some emotional moments, but lacking depth or sustained resonance.

    Score 4 - Good:
    - Readability: Strong; clean, largely error-free prose; vocabulary is varied and appropriate.
    - Structure: Engaging; well-defined narrative arc, generally even pacing, smooth transitions.
    - Character & Dialogue: Believable; characters feel real and dialogue sounds natural and purposeful.
    - Creativity: Inventive; fresh elements or ideas; at least one well-earned surprise.
    - Emotional Impact: Cohesive; consistent emotional resonance, themes develop clearly to a satisfying close.

    Score 5 - Excellent:
    - Readability: Flawless; elegant, polished prose with masterful command of grammar, style, and diction.
    - Structure: Masterful; seamless structure, perfectly tuned pacing, and enriching transitions.
    - Character & Dialogue: Fully realized; deep, nuanced characters, dialogue distinct and revealing.
    - Creativity: Original and bold; highly inventive storyline with inevitable yet startling surprises.
    - Emotional Impact: Profound; powerful emotional impact and strong thematic unity that leaves a lasting impression.

    If the text lacks substantial content or contains nonsensical or irrelevant elements, assign a score of 1.

    Respond with only a single integer: 1, 2, 3, 4, or 5"""

        user_message = f"""Story to evaluate:

    {response_text}"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
