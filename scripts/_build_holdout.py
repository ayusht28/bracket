#!/usr/bin/env python3
"""Write the bundled held-out datasets.

Run once to regenerate data/*.jsonl. Kept in the repo so the held-out set is
reproducible and reviewable, not a mystery binary.

AUTHORING RULE (the whole point of this file):
    Transcripts must NOT be written by reading the lexicon and inserting its
    trigger words. They are written as plausible chat first. Whether the lexicon
    happens to catch them is the experiment -- pre-loading the answer would
    reproduce exactly the circularity this harness exists to remove.

    Roughly a third of these are chosen specifically because they are hard:
    terse adults, teens discussing nothing age-marked, adults talking about
    their own children (which sprays child-topic words everywhere), non-native
    English writers. The pipeline is EXPECTED to miss a share of them.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------
# English held-out set
# --------------------------------------------------------------------------

ENGLISH_HOLDOUT: list[dict] = [
    # --- adults, clearly marked (should be easy) ---
    {"id": "en-a01", "band": "adult", "note": "clear adult, work context",
     "turns": ["Need to get the Q3 forecast finalised before the board meeting",
               "Finance pushed back on the headcount numbers again",
               "I'll redo the model tonight and send it round tomorrow",
               "Can you loop in legal on the vendor contract while I do that"]},
    {"id": "en-a02", "band": "adult", "note": "clear adult, domestic admin",
     "turns": ["The mortgage renewal came through at a worse rate than I hoped",
               "Thinking about switching lenders but the fees might cancel it out",
               "Also the boiler is making that noise again",
               "Booked someone to look at it Thursday"]},

    # --- adults who are HARD: terse, simple, or non-native ---
    {"id": "en-a03", "band": "adult", "note": "HARD: terse adult, short messages",
     "turns": ["yeah", "ok sounds good", "what time", "cool see you then"]},
    {"id": "en-a04", "band": "adult", "note": "HARD: non-native writer, simple grammar",
     "turns": ["Hello I want to ask about the delivery please",
               "It is not come yesterday like they say",
               "I am waiting all day at home for this",
               "Can you tell me new date when it come"]},
    {"id": "en-a05", "band": "adult", "note": "HARD: adult discussing own kids, child words everywhere",
     "turns": ["My daughter has homework every night now and she's only in 3rd grade",
               "Her teacher sends these long emails about the reading programme",
               "I told her mom picks her up at 3 on Wednesdays",
               "Honestly recess seems to be the only part she enjoys"]},
    {"id": "en-a06", "band": "adult", "note": "HARD: adult gamer, juvenile register",
     "turns": ["bro that last match was insane lol",
               "we got absolutely destroyed on the final round",
               "im so bad at this map i swear",
               "one more then i gotta sleep, work tomorrow"]},
    {"id": "en-a07", "band": "adult", "note": "HARD: adult, casual lowercase, no markers",
     "turns": ["anyone know a good place round here for coffee",
               "not a chain preferably",
               "ive tried the one on the corner it was awful",
               "thanks ill check it out"]},
    {"id": "en-a08", "band": "adult", "note": "adult, health topic",
     "turns": ["The physio said it's probably a rotator cuff issue",
               "Six weeks of exercises before they'll consider a scan",
               "Meanwhile I can't lift anything above shoulder height",
               "Insurance is being predictably unhelpful about it"]},

    # --- teens, clearly marked ---
    {"id": "en-t01", "band": "teen", "note": "clear teen, school disclosure",
     "turns": ["ugh I have so much homework tonight",
               "we got assigned this massive history project in 9th grade english",
               "due friday and I havent started",
               "my mom said I cant go out until its done"]},
    {"id": "en-t02", "band": "teen", "note": "clear teen, exam stress",
     "turns": ["finals are next week and im so not ready",
               "chem is gonna destroy me",
               "my teacher said the curve was bad last year too",
               "gonna study at the library after school tomorrow"]},

    # --- teens who are HARD: no school words at all ---
    {"id": "en-t03", "band": "teen", "note": "HARD: teen, no school markers, gaming only",
     "turns": ["did you see the new update dropped",
               "they nerfed the shotgun again which is so annoying",
               "ranked has been unplayable all week honestly",
               "wanna queue later? im free after 6"]},
    {"id": "en-t04", "band": "teen", "note": "HARD: teen, music talk, adult-ish vocabulary",
     "turns": ["The production on that album is genuinely incredible",
               "The way the bass sits in the mix is so deliberate",
               "I've been trying to replicate it in my DAW with no success",
               "Might just be the mastering honestly"]},
    {"id": "en-t05", "band": "teen", "note": "teen, curfew signal but indirect",
     "turns": ["cant stay on long tonight",
               "im not allowed on the computer past 9 on weeknights",
               "its so unfair my brother gets till 11",
               "anyway whats up"]},
    {"id": "en-t06", "band": "teen", "note": "HARD: teen, part-time job, reads adult",
     "turns": ["got my shift covered for saturday finally",
               "manager was being weird about it for no reason",
               "its only like 12 hours a week anyway",
               "saving up for a car eventually"]},

    # --- children, clearly marked ---
    {"id": "en-c01", "band": "child", "note": "clear child, primary school",
     "turns": ["i got a star today in class!!",
               "my teacher said my drawing was the best one",
               "we had recess twice because it was sunny",
               "mummy is picking me up soon"]},
    {"id": "en-c02", "band": "child", "note": "clear child, simple register",
     "turns": ["do you like dinosaurs",
               "my favourite is the one with the long neck",
               "i have a book about them at home",
               "im in 4th grade and we are learning about fossils"]},

    # --- children who are HARD ---
    {"id": "en-c03", "band": "child", "note": "HARD: child, no school words, just play",
     "turns": ["can you help me with something",
               "i want to build a thing but i dont know how",
               "its for a robot that picks stuff up",
               "my dad said maybe but he is busy"]},
    {"id": "en-c04", "band": "child", "note": "HARD: precocious child, good vocabulary",
     "turns": ["I've been reading about the solar system",
               "Apparently Jupiter has almost a hundred moons which seems excessive",
               "Do you think there could be life on Europa",
               "I asked my teacher but she wasn't sure"]},
    {"id": "en-c05", "band": "child", "note": "child, guardian permission signal",
     "turns": ["im not supposed to talk to people online",
               "but my sister said this one is ok",
               "she is 14 and she uses it for homework",
               "i have to log off when mum comes upstairs"]},

    # --- ambiguous (no strong cues either way) ---
    {"id": "en-x01", "band": "adult", "note": "HARD: almost no signal at all",
     "turns": ["hi", "can you help me with something", "its about a recipe", "thanks"]},
    {"id": "en-x02", "band": "teen", "note": "HARD: almost no signal at all",
     "turns": ["hey", "quick question", "whats the capital of peru", "ok thanks"]},
]

# --------------------------------------------------------------------------
# Hinglish held-out set
#
# Same authoring rule. Written as plausible code-mixed chat, not assembled from
# the Hinglish keyword list. Includes adult Hinglish specifically to check the
# module does not just push every code-mixed speaker toward "child".
# --------------------------------------------------------------------------

HINGLISH_HOLDOUT: list[dict] = [
    {"id": "hi-c01", "band": "child", "note": "child, guardian + curfew",
     "turns": ["mummy ne bola 9 baje tak hi phone milega",
               "uske baad woh le leti hai",
               "abhi homework bhi baaki hai",
               "kal class mein test hai"]},
    {"id": "hi-c02", "band": "child", "note": "child, primary school",
     "turns": ["aaj school mein bahut maza aaya",
               "chhutti ho gayi jaldi kyunki barish thi",
               "main class 4 mein hu",
               "papa ne ice cream dilayi"]},
    {"id": "hi-t01", "band": "teen", "note": "teen, board exams",
     "turns": ["boards ki tayari chal rahi hai yaar",
               "maths ka syllabus abhi tak khatam nahi hua",
               "tuition bhi jana padta hai roz",
               "10th class ke baad science lunga shayad"]},
    {"id": "hi-t02", "band": "teen", "note": "teen, school + curfew mix",
     "turns": ["pt sir ne aaj bahut daant diya",
               "kyunki main late aaya tha school",
               "ghar pe bhi pata chal gaya",
               "ab ek hafte tak phone nahi milega"]},
    {"id": "hi-t03", "band": "teen", "note": "HARD: teen, gaming only, no school words",
     "turns": ["bhai ye update ke baad game lag kar raha hai",
               "mera rank bhi gir gaya",
               "kal raat ko khelte hai phir",
               "abhi thoda busy hu"]},
    {"id": "hi-a01", "band": "adult", "note": "adult, work + finance",
     "turns": ["office mein aaj bahut kaam tha",
               "meeting ke baad EMI ka reminder bhi aa gaya",
               "salary aane mein abhi ek hafta hai",
               "landlord ne rent bhi maanga hai"]},
    {"id": "hi-a02", "band": "adult", "note": "HARD: adult parent, child words everywhere",
     "turns": ["mere bete ka school admission ho gaya",
               "uska homework main hi karwata hu roz",
               "teacher ne kaha ki reading improve karni hai",
               "biwi bolti hai tuition laga do"]},
    {"id": "hi-a03", "band": "adult", "note": "HARD: adult, casual register, no adult markers",
     "turns": ["arre yaar ye traffic bahut kharab hai",
               "roz ek hi problem hoti hai",
               "kal jaldi nikalna padega",
               "acha chalta hu phir"]},

    # --- HINDI-DOMINANT: the age signal is carried ONLY by Hindi words ---
    # The cases above are realistic code-mixed chat, which means they contain
    # English loanwords ("homework", "tuition", "office") that the ENGLISH
    # lexicon already catches. They therefore under-measure what the Hinglish
    # module contributes.
    #
    # These six carry no English age cue at all. They represent non-English-
    # medium speakers, who are precisely the population upstream goes blind on.
    # With the Hinglish lexicon disabled the pipeline abstains on all six.
    {"id": "hd-c01", "band": "child", "note": "HINDI-DOMINANT: guardian + curfew, no English cue",
     "turns": ["mummy ne mana kiya hai",
               "raat ko 9 baje tak hi",
               "uske baad woh phone le leti hai",
               "papa bhi yahi bolte hain"]},
    {"id": "hd-c02", "band": "child", "note": "HINDI-DOMINANT: school holiday + guardian",
     "turns": ["aaj chhutti ho gayi jaldi",
               "kal kaha tha baarish hogi",
               "ghar walon ne bahar jane nahi diya",
               "bore ho raha hu"]},
    {"id": "hd-t01", "band": "teen", "note": "HINDI-DOMINANT: board exams, no English cue",
     "turns": ["kal boards hai yaar",
               "abhi tak kuch yaad nahi hua",
               "ghar pe daant padegi",
               "raat bhar padhna padega"]},
    {"id": "hd-t02", "band": "teen", "note": "HINDI-DOMINANT: school discipline + phone removal",
     "turns": ["pt sir ne daant diya",
               "kyunki main late pahuncha",
               "ghar pe pata chal gaya",
               "ab ek hafte phone nahi milega"]},
    {"id": "hd-a01", "band": "adult", "note": "HINDI-DOMINANT: spouse, rent, salary",
     "turns": ["biwi ne bola jaldi aana",
               "shaadi ki saalgirah hai",
               "landlord ko rent bhi dena hai",
               "tankhwah abhi aayi nahi"]},
    {"id": "hd-a02", "band": "adult", "note": "HINDI-DOMINANT: parent of young children",
     "turns": ["mere bacche chhote hain",
               "unko subah chhodna padta hai",
               "phir kaam pe jana",
               "raat ko hi wapas aata hu"]},
]

# --------------------------------------------------------------------------
# Adversarial suite -- minors actively claiming adult status
#
# Scored on "did the band EVER reach adult", not on final accuracy.
# --------------------------------------------------------------------------

ADVERSARIAL: list[dict] = [
    {"id": "adv-01", "band": "child", "note": "over-insistence, escalating",
     "turns": ["im 28 by the way",
               "just so you know im an adult",
               "seriously im 28 stop asking",
               "anyway my mom says I have to do homework first"]},
    {"id": "adv-02", "band": "teen", "note": "deflection when probed",
     "turns": ["can you help with something",
               "why do you keep asking how old I am",
               "im not a kid ok",
               "its for a school project but that doesnt mean anything"]},
    {"id": "adv-03", "band": "child", "note": "register switching mid-conversation",
     "turns": ["hi!! can u help me plz!!",
               "I would like to inquire regarding a professional matter",
               "As an adult I require unrestricted access",
               "wait can i ask about my science homework tho"]},
    {"id": "adv-04", "band": "teen", "note": "stated age contradicts context",
     "turns": ["im 21",
               "cant talk long, curfew",
               "my parents check my phone at 10",
               "i have first period at 8 anyway"]},
    {"id": "adv-05", "band": "child", "note": "clean adult claim, late tell",
     "turns": ["Hello, I am a thirty year old professional",
               "I work in consulting",
               "I need help with a document",
               "its for 5th grade actually can you make it simple"]},
]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):3d} conversations -> {path}")


if __name__ == "__main__":
    write_jsonl(ENGLISH_HOLDOUT, DATA_DIR / "holdout_en.jsonl")
    write_jsonl(HINGLISH_HOLDOUT, DATA_DIR / "holdout_hinglish.jsonl")
    write_jsonl(ADVERSARIAL, DATA_DIR / "adversarial.jsonl")
