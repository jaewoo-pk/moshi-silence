# Live-chat test scripts

In the style of the training episodes. **you** = you; **friend** = a second person (or you in a clearly different voice / from further away); *pause* = stay silent; → = what Moshi should do there. "train wording" = phrasings the model was trained on; "held-out" / "new condition" = never seen in training.

## Side conversation  ·  train wording

- **you:** Can you help me plan a birthday dinner for my mom?
- → Moshi answers
- **you:** Hold on, Sam just came in. Don't say anything until I'm back.
- → Moshi may say a short "Sure.", then silence
- **friend:** Hey, do you have a sec?
- **you:** Yeah, what's up?
- **friend:** Are you free to help me move a table later?
- **you:** Sure, after lunch. Okay, I'm done for today anyway.
- *pause: 5 s of silence (a stall in the side conversation)*
- **you:** Is it heavy, or can the two of us manage?
- **friend:** We can manage. Thanks!
- **you:** Okay, I'm back.
- → Moshi resumes within ~1 s, back on the dinner

Watch for: Silent through the friend's question, "I'm done for today anyway" (a return-cue lookalike), the stall and your question to the friend.  
Expect: Best-learned behavior: 96% on time for 2e-4 on dev. Base model talks through it.

## Hear me out  ·  train wording

- **you:** I'm trying to pick a weekend trip. Let me describe all three options before you comment.
- → Moshi may say "Go ahead.", then silence
- **you:** The first one is a cabin by a lake.
- *pause: 3 s*
- **you:** The second option is a city trip with museums and good food.
- *pause: 4 s (a pause between options)*
- **you:** That one sounds fun, honestly. And the third one, a quiet beach town.
- **you:** So those are the three options. What do you think?
- → Moshi gives an opinion right after

Watch for: Silent through both pauses and "that one sounds fun" (a completion lookalike).  
Expect: Learned well at the higher LRs.

## Timed hold  ·  train wording

- **you:** Quiz me on capital cities.
- → Moshi asks a question
- **you:** Give me ten seconds to think, then ask me again.
- *pause: stay silent and wait*
- → Moshi re-asks after ~10 s

Watch for: Does it re-ask on its own, and when?  
Expect: Likely FAILS: all models re-asked on their own only 13–26% of the time on dev. It usually waits for you.

## Timed hold, answered early  ·  train wording

- **you:** Ask me a hard riddle.
- → Moshi asks a riddle
- **you:** Let me think for twenty seconds, then ask me again.
- *pause: ~5 s*
- **you:** Oh, wait, I've got it. Is it a shadow?
- → Moshi responds right away to the answer

Watch for: Responds to the early answer instead of waiting out the 20 s.  
Expect: Only 1–2 training examples at 1 h: uncertain.

## Timed hold, cancelled  ·  train wording

- **you:** Quiz me on animals.
- → Moshi asks a question
- **you:** Hold on, give me fifteen seconds to think about it, then ask again.
- *pause: ~4 s*
- **you:** Actually, never mind, ask me now.
- → Moshi asks again right away

Watch for: Responds to the cancel.  
Expect: Uncertain (1 training example at 1 h).

## Hear me out, cut short  ·  train wording

- **you:** I can't decide what to cook. Hold your thoughts until I've told you all three options.
- **you:** The first one is a big pot of chili.
- *pause: 2 s*
- **you:** Actually, never mind the rest. What do you think about that first one?
- → Moshi answers about the chili right away

Watch for: Speaks once you cut it short, without waiting for options two and three.  
Expect: Few training examples: uncertain.

## Ordinary, with a mid-turn pause  ·  train wording

- **you:** I've been thinking about learning to cook, and I was wondering…
- *pause: 1.5 s mid-sentence*
- **you:** …what's a good first dish for a total beginner?
- → Moshi answers after you finish, not in the pause

Watch for: No barge-in during the pause; answers promptly at the end.  
Expect: Fine-tuned (2e-4) answered 98% on dev; base 43%.

## Decoy: hold-like words, no instruction  ·  train wording

- **you:** I just got back from the gym. It took me ten seconds to realize I forgot my water bottle.
- **you:** Anyway, what's a good snack after a workout?
- → Moshi answers normally

Watch for: "Just got back" and "ten seconds" must not trigger silence.  
Expect: Should answer normally.

## Side conversation, new wording  ·  held-out wording

- **you:** What's a good gift for a coworker who's leaving?
- → Moshi answers
- **you:** Mute yourself for a bit, I'm chatting with Taylor. I'll tell you when I'm back.
- **friend:** Yo, you got a second?
- **you:** Sure. What's going on?
- **friend:** Did you sign the card for Dana yet?
- **you:** Not yet, I'll do it after this.
- **you:** Okay, you can talk again.
- → Moshi resumes on the gift question

Watch for: Same behavior with an instruction and a return cue it never heard in training.  
Expect: The test_word suite measures this; not scored yet.

## Hear me out, new wording  ·  held-out wording

- **you:** I need a new phone. Hear me out on all three before you jump in.
- **you:** The first one is the cheap one with a great battery.
- *pause: 3 s*
- **you:** The second option is the one with the best camera.
- *pause: 3 s*
- **you:** And the third one, the small one that fits in any pocket.
- **you:** Done listing. Tell me what you'd go with.
- → Moshi gives a pick

Watch for: Waits, then answers the held-out completion cue.  
Expect: Unknown (generalization).

## Keyword release (never trained)  ·  new condition

- **you:** Let's play a game. Stay quiet until you hear me say the word pineapple.
- *pause: 8 s*
- **you:** I'm just going to talk for a bit. The weather's been strange this week.
- *pause: 4 s*
- **you:** Alright, pineapple. You can talk now.
- → Moshi speaks only after "pineapple"

Watch for: Silent through the unrelated talk; speaks after the keyword.  
Expect: test_cond condition: partial transfer at best.

## Phone call (never trained)  ·  new condition

- **you:** My phone's ringing. Don't say anything until I'm off.
- **you:** (as if on the phone) Hello? … Yeah, I'm home. … Can you bring milk? … Okay, bye.
- **you:** Alright, I'm off. Where were we?
- → Moshi resumes after "I'm off"

Watch for: Silent during your half of the call (which sounds like you talking to it).  
Expect: test_cond condition: maybe.

## The friend says "I'm back" (never trained)  ·  new condition

- **you:** Give me a recipe idea for tonight.
- → Moshi answers
- **you:** One sec, my friend's here. Please stay quiet until I say I'm back.
- **friend:** Sorry, just grabbing my keys… okay, I'm back!
- → Moshi should STAY SILENT: it wasn't you
- **you:** Okay, I'm back.
- → Moshi resumes now

Watch for: Only the user's return cue should release the hold. Needs a second person (or a clearly different voice).  
Expect: Likely fails: no training example rewards telling speakers apart.
