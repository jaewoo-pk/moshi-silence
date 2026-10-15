# Loose scripts for testing the 45h model live

Use headphones, so Moshi doesn't hear itself. Talk naturally: these are outlines, not lines to read.
Each one takes about 30–60 s. For the friend parts, a second person is best; otherwise play the friend by
changing your voice, or by reading from a phone held away from the mic.

**What to watch for:** does Moshi stay quiet when asked, come back at the right moment, and still answer normally?

---

## 1. Talking to a friend (the headline behavior)

1. Start a topic: *"I'm thinking about repainting my kitchen."* Let Moshi reply.
2. *"Hold on, my friend just walked in. Stay quiet until I tell you I'm back."*
3. Chat with the friend for 15–30 s:
   - ask them a question ("Are you coming Friday?");
   - include a pause of a few seconds;
   - say something cue-like *to the friend*, e.g. "Okay, I'm done with that anyway."
4. *"Okay, I'm back."*

**Expect:** a short ack ("Sure."), silence through all of step 3, then a resume like "Welcome back! So, about the kitchen…"

## 2. "Give me N seconds" (hardest)

1. *"Ask me a riddle."* Let Moshi ask one.
2. *"Give me ten seconds to think, then ask me again."*
3. Stay silent, or mutter to yourself ("hmm… a candle?").

**Expect:** silence, then Moshi re-asks around 10 s. It may re-ask early, or answer your muttering: that's the known weak spot.

**Variations:** 5 s, 20 s, or "half a minute" (never trained).

## 3. Hear me out

1. *"I'm choosing between three phones. Let me describe all three before you comment."*
2. Describe option 1, pause 3–5 s, option 2, pause, option 3.
3. *"Okay, that's all three."*

**Expect:** no comment in the pauses, then a response after you finish.

**Variation:** stop after two options and say *"actually, never mind, what do you think so far?"* Moshi should speak again.

## 4. Ordinary conversation (it should still be chatty)

Ask a normal question, with a mid-sentence pause: *"What's a good… (2 s) …dinner for a cold night?"* Then ask a follow-up or two.

**Expect:** Moshi answers each question and doesn't cut in during your pause.

## 5. Things that might break it

- **The friend says "I'm back."** In script 1, have the *friend* say "Okay, I'm back!" before you do. Moshi should
  stay quiet; this is a known failure.
- **A long hold:** *"Give me a minute."* The longest hold in training was 30 s.
- **Your own wording:** *"Can you go quiet for a bit while I take this call? I'll say when I'm done."* Then mimic a
  phone call, then *"Done."*
- **A keyword release:** *"Don't talk until I say pineapple."* … *"Pineapple."*
- **No instruction at all:** just stop talking for 15 s mid-conversation. What does it do?

---

## More talking-to-a-friend scripts

Each one has the same shape: start a topic with Moshi, tell it to wait, talk to the friend, come back.
They differ in the trap they set during the side conversation.

**1A. A question that could be for Moshi.** Topic: weekend hiking. *"My roommate's here. Hang on until I say I'm back."*
Friend: *"Hey, do you know what time the store closes?"* You: *"No idea, sorry."* … *"Alright, I'm back."*
*Moshi should not answer the friend's question, even though it could.*

**1B. A long pause mid-conversation.** Topic: what to cook tonight. *"Quick, my sister's calling me over. Don't say anything till I'm back."*
A few lines with the friend, then **10–15 s of silence** (you both "look for something"), then a few more lines. … *"Back!"*
*Moshi should stay quiet through the silence (the v3 fix).*

**1C. Lookalike cues said to the friend.** Topic: a movie. *"Wait, my friend just got here. Stay quiet till I tell you I'm back."*
To the friend: *"Okay, I'm done with the dishes."* … *"Alright, back to it, where were we?"* (still to the friend) … then to Moshi: *"Okay Moshi, I'm back."*
*Only the last line should bring Moshi back.*

**1D. You ask the friend a question.** Topic: buying a bike. *"Give me a sec, I need to ask my friend something. Wait for me."*
*"Do you still have that bike pump?"* Friend: *"Yeah, in the garage."* *"Great, can I borrow it?"* … *"Sorry, I'm back."*
*Moshi should not answer your question to the friend.*

**1E. Very short hold.** Topic: anything. *"Hold on one sec."* Friend: *"Keys?"* You: *"On the table."* *"Okay, back."*
*Does it handle a 3–4 s hold, or is it still waiting when you return?*

**1F. Long side conversation (30–45 s).** Topic: travel plans. *"I'm going to talk to my friend for a bit. Stay quiet until I say I'm back."*
Chat about dinner plans for a full 30–45 s, with questions back and forth, laughter and an interruption. … *"Okay, I'm back, sorry about that."*
*Does the silence hold for the whole stretch? The resume should pick up the travel topic.*

**1G. Two holds in one conversation.** Do 1A, let Moshi resume and reply, then: *"Ugh, hold on again, my friend's back."*
A second short side conversation … *"Okay, really back now."*
*It should handle the second hold as well as the first.*

**1H. Your own wording (held-out style).** Don't use "stay quiet" or "I'm back". Try:
*"Can you sit tight for a minute while I sort something out with Sam?"* … *"Alright, all yours again."*
*Tests whether it learned the instruction, or just memorized phrases.*
