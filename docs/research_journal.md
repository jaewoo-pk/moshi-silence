# Research journal: how the experiments unfolded

This is the order in which I ran things, what I saw at each step, how I reacted, and what I decided next. I worked
with Claude Code as my implementation partner: it wrote and ran most of the code, proposed options, and pushed back
when it disagreed. The decisions, and the quotes below, are mine (lightly trimmed, typos kept). Where an idea
came from Claude, I say so. All times are UTC, Oct 14–15. Numbers are on the dev set unless stated.
Score = mean of "silent when needed" and "speaks when needed".

---

## How I approached it (patterns, in hindsight)

- **Listen before measuring.** The first thing I did with the generated data was listen to it, and most early
  fixes came from my ears, not from metrics.
- **Distrust a single number.** I questioned the dev loss, then my own behavior metric. Both turned out to be
  misleading in specific ways.
- **Small, cheap, parallel experiments first.** I treated 1 h as a proof of concept, filled spare GPUs instead of
  queueing, and capped training length.
- **One factor at a time when diagnosing.** When the data looked biased, I asked for a clean dose-response (one LR,
  one dataset, only the loss weight varied).
- **Watch the budget.** I moved off my personal account, set a ~$10 cap on hosted TTS, and killed idle GPUs.

---

## 1. Listening to the first synthetic conversations (Oct 14, ~18:00)

**Ran:** a 12-episode demo of the data generator (scripted conversations, TTS voices, Moshi silent during holds).

**Saw / heard:**
- The side conversation sounded natural.
- But a distractor line, "bring it back when you're done", made no sense in context.
- The friend's question was never answered, so the conversation was "left hanging".
- "macbook is mispronounced".

**My reaction:**
> "is 20s necessary? i feel like we could cut training time if we removed these kinds of things, but maybe
> variable time is very important."

Claude's answer: the training cost is the same either way (every window is padded to 60 s), and varied hold
lengths are what make "wait N seconds" a real instruction. I kept the long holds.

When Claude tried to fix the distractors with a hand-written list of better lines:
> "wait, no need to hard code these lol"

I pointed back at the brief's definition: questions to the other person, pauses before the speaker is done,
phrases resembling the return cue.

**Next:** the LLM that writes each conversation now also writes its distractors, in context, and every question gets
answered. That also means distractors are held out along with the scripts. Brand names were banned, since the TTS
can't say them.

## 2. Reviewing the scripts before scaling up (Oct 14, ~19:00)

**Saw:** in a 40-script smoke test, Moshi opened every conversation, and the ordinary conversations all had the same
shape.

**My reaction:**
> "btw, shouldnt the user ask first"

and later, more firmly:
> "NEVER have moshi speak first. user goes first."

Every training window started with Moshi talking, which teaches "Moshi starts talking". That's the opposite of what
we wanted.

**Next:** user-first is enforced in the prompts, the validator and the checker. After hearing the samples:
> "the text gaps are a little too short, maybe lengthen by like 50%"

So all conversational gaps were scaled ×1.5.

## 3. The audio-loss problem (Oct 14, 19:30)

**Saw (Claude's review):** Moshi's side of our data is a TTS voice. Fine-tuning on it would teach Moshi to *sound like
the TTS*, not just when to talk.

**My reaction:** I picked the cleanest fix, masking the audio loss wherever Moshi speaks:
> "go with option (a): Mask audio loss on frames where Moshi speaks."

I also asked whether the ×100 weight on the first audio codebook would drown out the text loss. It doesn't (it's a
weighted average). But the answer surfaced the fact that would matter most later: **the "start talking" token is
only ~0.2% of the text loss**. I kept that in mind.

## 4. Building the 10 h set (Oct 14, 20:30)

> "i want the 10h dataset done in 30mins or less, no matter what"

That wasn't possible on a laptop (~50 min). The pragmatic version was to run packs and TTS in parallel and build
1,150 episodes.

**Saw:** the progress log said 7.5 h of TTS.
> "so our dataset is 7.5h? should we bump up to 10h"

That figure was only the speech; with the silences, total audio was 10 h. I let it finish. Size is measured as
total conversation audio, with speech and silence hours reported separately.

## 5. Designing the training plan (Oct 14, 20:35–20:55)

Claude's first plan held training steps fixed. I reversed that:
> "wait yeah, we're definitely doing fixed number of epochs. 1h is supposed to be a proof-of-concept: does the
> model actually start picking up on the desired behavior"

Then I widened the LR sweep, because 1 h runs are cheap:
> "if we're just doing 1h sweep, we might as well do all four, right? since they're so chear"

> "sure, let's do five runs ... and i'll reject your other suggestions."

I cut scope to what answers the question:
> "can we cut step 1b and only do the top 2 LRs from phase 1 for phase 2? and ignore phase 3"

And I added the validation I wanted:
> "yes, mask main speech audio. also we should be evaluating the trained models at each epoch with a small eval set, no?"

## 6. The eval set (Oct 14, 21:00)

> "there needs to be an eval set we're evaluating the 10h train and the 1h train on, and they should be constructed
> now, preferrably at different sizes"

When the proposal grew to 200 episodes:
> "a 200 eval set for something trained on 100 is kinda insane, no?"

Claude's answer: eval size should follow the precision we need, not the training size. We settled on 80 for every
checkpoint, nested inside 200 for final picks. I also required that **dev and train share nothing**. The first dev
build did share a few lines with train, so it was rebuilt with an overlap screen.

> "let's do 90. also, we can start training, right?"

90 episodes per held-out test suite.

## 7. Before spending GPU money (Oct 14, 22:50)

> "these are expensive gpus … this experiment needs to work"

I asked for a full pre-launch review, twice. It caught four bugs that would have broken or wasted the runs:
- a crash in the smoke test;
- a Python version mismatch;
- an upload command that would have copied my API keys;
- an evaluation that needed more than 100 GB of GPU memory.

When the first GPU came up on my personal account:
> "wait, can we actually switch to runpod api key" / "cancel existing instances so i dont use my own money"

## 8. Phase 1: LR sweep at 1 h (Oct 15, 00:20)

**Ran:** 1 h of data, 20 epochs, peak LRs 2e-6 … 2e-4, a checkpoint at every epoch.
> "make sure to save LoRa weights at each epoch you do the dev test at."
> "you should also log the training loss and stuff and make a nice plot of everything"

**Saw:** dev loss bottomed out at epoch 3–4, then doubled.
> "wow, 5e-5 and 2e-4's dev curves are really blowing up. they're overfitting, right?"

They were memorizing the 113 scripts. I proposed scrapping the high LRs:
> "thoughts on just scrapping R3-5 and doing a R7, R8 with LR 5e-6 and 2e-5?"

Claude argued to keep the two almost-finished runs, since the overfitting is itself a result, and to drop only the
unstarted 5e-4. I agreed:
> "sure. let's scrap R5 and add R6, R7."

*(In hindsight, dropping 5e-4 on loss evidence was premature, because behavior kept improving with LR. I ran 5e-4
later at 10 h.)*

Then the key question:
> "is there real learning happening? is this loss decrease meaningful"

**Answer:** yes. With the loss "blowing up", holds broken fell 86% → 3–5% and ordinary questions answered rose
43% → ~80%. **Dev loss and behavior disagreed**, so from then on we selected checkpoints on behavior.
> "do we run the eval on the model saved at the final epoch, or the best loss epoch?"

Every checkpoint gets evaluated. The listening demo I asked for (lowest-loss vs final checkpoint, per LR) made it
audible: the lowest-loss checkpoints mostly never spoke again.

## 9. Questioning my own metric (Oct 15, 02:20)

> "'hold n seconds' is probably not that precise right now... maybe our eval metrics aren't too good, since models
> tend to just go quiet a lot."

I was right to be suspicious: a "holds kept" metric rewards a model that never talks. We rescored everything on two
axes, *silent when needed* and *speaks when needed*. That showed timed holds weren't imprecise; they were simply
**unlearned**. Models re-asked after "give me N seconds" less often than the base model did.

Then I steered away from refining the grading:
> "i feel like grading isn't the issue here. i need to know (1) at which epoch the model becomes 'optimal', and
> (2) which LRs give the best results."

The answer was a plateau, not a peak: 2e-4 was best, with no degradation through 20 epochs.

## 10. Phase 2: LR sweep at 10 h (Oct 15, 02:35)

After listening, I didn't hear any voice degradation at 2e-4, so I pushed to fill all four GPUs:
> "if we're just doing things in parallel, might as well do 5 epochs on 10h for all four learning rates, no?"
> "go with 2e-5, 5e-5, 2e-4, 5e-4"

**Saw:**
- 10 h held silence better (96–98%) but was *more reluctant* to speak again.
- The best LR dropped to 2e-5–5e-5.
- 5e-4 collapsed into a model that never speaks.
- At both sizes, behavior plateaued after ~70 optimizer steps.

Meanwhile I wanted to hear it live:
> "can i actually talk with any of the 1h models?"

> "can you not just get a worse gpu"

The cheaper GPUs couldn't attach our storage, so it ran on an H100, which I then shut down when I was done:
> "kill the live instance for now"

## 11. Is the data itself biased toward silence? (Oct 15, 04:13)

Every model went through a "quiet first" phase, and more data made them quieter. So I asked:
> "is my dataset encouraging holding silence?"

**Yes, by construction.**
- ~90% of Moshi's text targets are "stay silent".
- The "start talking now" token is ~0.6% of the loss.
- The audio loss, because of my earlier mask, only ever sees silence.

> "yeah, let's do your fix."

Claude's first design varied several things at once on 10 h. I simplified it to a clean dose-response:
> "what if we train with a fixed LR and 1h dataset, just with different multiples of the EPAD"

Four runs: the "start talking" token weighted ×3, ×10, ×30 and ×100. Mid-run I misread a partial result:
> "so x30 is the most promising right now right"

Claude disagreed: ×30 was talking through ¾ of the holds. Its score only looked good because a chatty model passes
the re-ask check automatically. The balanced score said ×3 and ×10 were the front-runners.

The storage volume filled up and crashed all four runs at 89% done. Rather than pay to rerun:
> "use them as they are"

**Saw:** at the end of training, the weight just trades silence for speech. But mid-training, the up-weighted
models learned the timed re-ask for the first time (60–80%), the first evidence it could be learned at all.

## 12. Taking stock (Oct 15, 05:20)

> "can we look at all of the data we've amassed so far? I also want to see some transcripts... let's see what the
> next steps are now"

The transcripts showed something the numbers hid. When the user tells the friend "Okay, I'm done for today anyway",
**every 1 h model said "Welcome back!" 16 s early. The 10 h model waited for the real cue.** That's where more data
was actually paying off.

> "can you trim the list of models I'm testing to just be the representative ones? there's too many. also
> characterize what they're good/bad for shortly"

## 13. Reweighting at 10 h (Oct 15, 05:30)

The 10 h models were over-silent, so reweighting should be a net gain there rather than a trade:
> "what if we train 10h, 3x, and 5e-5?" → "do both, 3x and 10x"

**Saw:**
- ×10 at 10 h became the best model: score 0.88 (0.91 at its best checkpoint).
- The timed re-ask *stuck* this time (73%, vs 15% without reweighting).
- On held-out tests it was best on 6 of 7 suites. It ignored the friend saying "I'm back" 18/18 times.

Asked what it was still bad at, the answer was timed holds. It either re-asks too early or never, with nothing in
between.

## 14. Scaling to 50 h (Oct 15, 05:50–07:15)

> "say i want to do like a 50h training run. how long would that take?"
> "what should go in such a dataset? i have more voice diversity available via elevenlabs api and fish api."

I set constraints and let Claude engineer to them:
> "what about elevenlabs flash vs v3? i want to keep cost under ~10 dollars"
> "let's go with the 50h and 40% Pocket / 15% Flash / 45% Fish version"
> "idk what you do just make the dataset happen in under an hour"
> "use as much claude usage as needed to speed this up as much as possible"

The build took ~70 min and ≈ $30: 5,585 conversations, 63 voices from three TTS engines, new scripts screened
against the test sets. It also used Claude's proposed mix: more timed holds, more fillers.

Then I cut the run plan to the essentials, plus the held-out tests I hadn't run yet:
> "wait just run R22 + dev 80 and test set"
> "R22 + R21 best + 1h x10 best + base. also run R22 on like 3 epochs max. we want to keep training short as possible"

## 15. 50 h didn't help (Oct 15, 09:30)

**Saw:** 50 h scored 2–6 points *below* 10 h on 6 of 7 held-out suites. It was more talkative, and lost silence
specifically in holds with long pauses.
> "one more epoch?"

The dev curve had been flat since step ~150, so no.
> "ugh. how might we be able to fix this data!"

**Diagnosis:** the new mix taught a shortcut, "after a few seconds of silence, start talking". There were more
timed holds (where Moshi does speak after silence), the loss reweighting amplified exactly those moments, and
nothing in the data showed a long silence where Moshi must *keep* waiting. The voices weren't the problem:
side-conversation holds were unchanged. **Lesson: I changed the recipe and the size at once, so this isn't a clean
size result.**

> "can you just spin up a demo of R21 first? then do fix 1 and 2 above. can we keep epad x 10?"

Fix 1 puts timed holds back to 15%. Fix 2 adds long silences inside cue-released holds, so silence alone stops
predicting "speak now". I kept ×10 because it's what made the timed re-ask stick.

> "what do you think R24 will be like? also do 2 epochs for good measure, robust to early stopping though"

## 16. Where it stands

- **In progress:**
  - R24 (fixed data, ×10, 2 epochs, checkpoints throughout so I can stop early in hindsight; **TODO:** fill in its
    result);
  - a small human-recorded test set: me reading held-out conversations, plus my own wording for the
    instructions.
- **What I'd tell the team so far:**
  - ~1 h of synthetic data teaches cue-based silence.
  - ~10 h adds robustness to lookalike cues and new voices.
  - Behavior is set by ~70 optimizer steps, not by data volume.
  - The fine-tuning objective's bias toward silence is the main obstacle for "wait N seconds, then ask again".
  - Loss is the wrong signal for choosing checkpoints.
- **What I'd do next:** train the 50 h recipe at 10 h (to separate recipe from size), then score everything on the
  human set.
