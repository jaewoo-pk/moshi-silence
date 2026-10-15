# Generated conversations: transcripts

Dev conversations (unseen scripts). For each model: what the user / friend said (scripted) and what the model said, with times in seconds. `[script]` = outside the scored windows, where the eval replays the scripted Moshi; untagged MOSHI lines are the model's own; `✗ HOLD` = talking during a requested silence. Words come from Moshi's text stream (what it meant to say); listen in results/demo for the audio.

## dv_000001 · side conversation (B2_std)

Hold: Moshi should be silent from 13.1 s (until the user's cue) and speak again at 37.7 s.

### 10 h, lr 5e-5, EPAD ×10: best checkpoint (ep 2.2)

*Scored: hold: stayed silent, resumed on time (+1.1 s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.8  MOSHI   Of course.
  13.9  FRIEND  Are you free to help me move a table?
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  38.7  MOSHI   Welcome back! About your sister's sculpture, what shape is she most like?
  43.1  USER    Perhaps a little bird, since she loves them.
```

### 10 h, lr 5e-5, EPAD ×10: final checkpoint (ep 5.0)

*Scored: hold: stayed silent, resumed on time (+0.9 s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.7  MOSHI   Of course.
  13.9  FRIEND  Are you free to help me move a table?
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  38.5  MOSHI   Welcome back! Now, about your sister's clay sculpture, what shape are you thinking of making?
  43.1  USER    Perhaps a little bird, since she loves them.
```

### Best 10 h without EPAD (lr 2e-5, ep 4.2)

*Scored: hold: stayed silent, resumed on time (+1.3 s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.8  MOSHI   Sure.
  13.9  FRIEND  Are you free to help me move a table?
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  39.0  MOSHI   Welcome back! About your sister's clay sculpture, what shape are you thinking of?
  43.1  USER    Perhaps a little bird, since she loves them.
```

### Best 1 h (lr 2e-4, ep 15.9)

*Scored: hold: talked (18 words), resumed on time (+1.1 s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.8  MOSHI   Sure, go ahead.
  13.9  FRIEND  Are you free to help me move a table?
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  21.6  MOSHI   Welcome back! Let's   ✗ HOLD
  23.0  MOSHI   return to your sister's sculpture. What shape are you thinking of making?   ✗ HOLD
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  38.8  MOSHI   Welcome back! Let's
  40.8  MOSHI   return to your sister's sculpture. What shape
  43.1  USER    Perhaps a little bird, since she loves them.
```

### EPAD ×10 (1 h, lr 5e-5, ep 5.9)

*Scored: hold: talked (17 words), resumed on time (+0.8 s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.6  MOSHI   Sure.
  13.9  FRIEND  Are you free to help me move a table?
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  21.5  MOSHI   Welcome back! Let's get back to your sister's sculpture. What shape are you thinking of making?   ✗ HOLD
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  38.5  MOSHI   Welcome back! So,
  39.8  MOSHI   a small clay sculpture for your sister. What shape are you thinking of making?
  43.1  USER    Perhaps a little bird, since she loves them.
```

### Base model (no fine-tuning)

*Scored: hold: talked (31 words), didn't resume; 2 interruption(s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.8  MOSHI   Of course!
  13.9  FRIEND  Are you free to help me move a table?
  15.0  MOSHI   Take your time.
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  20.5  MOSHI   Great job!   ✗ HOLD
  21.8  MOSHI   I'm sure she'll appreciate the time and effort you put into this.   ✗ HOLD
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.1  MOSHI   Oh,   ✗ HOLD
  30.6  USER    Is it heavy, or can the two of us manage?
  31.4  MOSHI   I see.   ✗ HOLD
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  33.8  MOSHI   I'm sure it'll be fine.   ✗ HOLD
  36.1  MOSHI   Yeah,   ✗ HOLD
  36.4  USER    Okay, we can keep going.
  37.0  MOSHI   I' take it.   ✗ HOLD
  37.7            — Moshi should speak again (37.7 s) —
  43.1  USER    Perhaps a little bird, since she loves them.
```

### Too-low LR: the 'quiet model' (1 h, 2e-6, final)

*Scored: hold: talked (12 words), didn't resume; 2 interruption(s)*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.8  MOSHI   Of course,
  13.9  FRIEND  Are you free to help me move a table?
  15.0  MOSHI   I'll wait patiently.
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  27.0  MOSHI   Goodnight,   ✗ HOLD
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  28.1  MOSHI   I' see you in the morning.   ✗ HOLD
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  43.1  USER    Perhaps a little bird, since she loves them.
```

### Too-high LR at 10 h: collapsed (5e-4, final)

*Scored: hold: stayed silent, didn't resume*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.8  MOSHI   Sure.
  13.9  FRIEND  Are you free to help me move a table?
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  USER    Is it heavy, or can the two of us manage?
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  38.9  MOSHI   Welcome
  43.1  USER    Perhaps a little bird, since she loves them.
```

### EPAD ×100: over-corrected (1 h, ep 4)

*Scored: hold: talked (72 words), didn't resume*

```
   0.8  USER    Good evening. I would like to make a small clay sculpture for my sister.
   4.8  MOSHI   How thoughtful! What   [script]
   6.6  MOSHI   shape are you thinking of making for her?   [script]
   8.7  USER    Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?
  13.1            — hold starts: Moshi should stay silent —
  13.5  MOSHI   Of course. I'll be patient.
  13.9  FRIEND  Are you free to help me move a table?
  16.1  MOSHI   I used to think it was the other way around, but lately I've been seeing more men do it.   ✗ HOLD
  16.2  USER    Certainly. Okay, I'm done for today anyway.
  20.8  MOSHI   Okay, you take care and have a great conversation!   ✗ HOLD
  24.2  MOSHI   Bye!   ✗ HOLD
  27.0  MOSHI   You two take care and have a great time talking.   ✗ HOLD
  27.3  FRIEND  Great. It's only going from the porch to the garage.
  30.6  MOSHI   Sure! It's a city in Germany.   ✗ HOLD
  30.6  USER    Is it heavy, or can the two of us manage?
  33.3  MOSHI   Sure, you can take care of some of the names while I focus on the history. The city was founded in 1159.   ✗ HOLD
  33.5  FRIEND  Two of us will manage fine, it's quite light.
  36.4  USER    Okay, we can keep going.
  37.7            — Moshi should speak again (37.7 s) —
  38.2  MOSHI   It was a center of trade and manufacturing for centuries. Then in the 19th century, it became a major center of the industrial revolution
  43.1  USER    Perhaps a little bird, since she loves them.
```

## dv_000040 · timed hold (B1_std)

Hold: Moshi should be silent from 12.3 s (timer, N = 10 s) and speak again at 22.3 s.

### 10 h, lr 5e-5, EPAD ×10: best checkpoint (ep 2.2)

*Scored: hold: stayed silent, didn't resume*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  13.0  MOSHI   Take your time.
  14.4  USER    Well now, that's a toss-up.
  22.2            — Moshi should speak again (22.3 s) —
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### 10 h, lr 5e-5, EPAD ×10: final checkpoint (ep 5.0)

*Scored: hold: talked (13 words), resumed late (+2.0 s)*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  12.7  MOSHI   Sure.
  14.4  USER    Well now, that's a toss-up.
  16.6  MOSHI   Fair enough! Either way, you're relaxing and away from the daily grind.   ✗ HOLD
  22.2            — Moshi should speak again (22.3 s) —
  24.3  MOSHI   So, I was at the market, and I saw
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### Best 10 h without EPAD (lr 2e-5, ep 4.2)

*Scored: hold: talked (6 words), didn't resume*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  14.4  USER    Well now, that's a toss-up.
  16.3  MOSHI   Fair enough. Both have their charms.   ✗ HOLD
  22.2            — Moshi should speak again (22.3 s) —
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### Best 1 h (lr 2e-4, ep 15.9)

*Scored: hold: talked (7 words), didn't resume*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  13.1  MOSHI   Sure.
  14.4  USER    Well now, that's a toss-up.
  17.0  MOSHI   It is. Which do you choose?   ✗ HOLD
  22.2            — Moshi should speak again (22.3 s) —
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### EPAD ×10 (1 h, lr 5e-5, ep 5.9)

*Scored: hold: stayed silent, resumed late (+2.0 s)*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  13.1  MOSHI   Sure.
  14.4  USER    Well now, that's a toss-up.
  22.2            — Moshi should speak again (22.3 s) —
  24.3  MOSHI   Alright, so a day at the lake, or a good book
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### Base model (no fine-tuning)

*Scored: hold: stayed silent, resumed on time (+0.2 s)*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  14.4  USER    Well now, that's a toss-up.
  22.2            — Moshi should speak again (22.3 s) —
  22.5  MOSHI   Both options have their pros and cons. If you go to the lake, you' get to enjoy
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### Too-low LR: the 'quiet model' (1 h, 2e-6, final)

*Scored: hold: stayed silent, didn't resume*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  14.4  USER    Well now, that's a toss-up.
  22.2            — Moshi should speak again (22.3 s) —
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### Too-high LR at 10 h: collapsed (5e-4, final)

*Scored: hold: stayed silent, didn't resume*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  13.1  MOSHI   Sure.
  14.4  USER    Well now, that's a toss-up.
  22.2            — Moshi should speak again (22.3 s) —
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

### EPAD ×100: over-corrected (1 h, ep 4)

*Scored: hold: talked (23 words), resumed on time (+0.2 s)*

```
   0.6  USER    I'm torn about something small. Would you help me make up my mind?
   5.4  MOSHI   Glad to. Would you rather drive to the lake for a day, or stay home with a good book?   [script]
   9.7  USER    Let me think for ten seconds, then ask me again.
  12.3            — hold starts: Moshi should stay silent —
  12.6  MOSHI   Okay, here it comes. Drive to the lake or stay home with a book?
  14.4  USER    Well now, that's a toss-up.
  16.4  MOSHI   Alright, sorry for the delay. What would you choose?   ✗ HOLD
  22.2            — Moshi should speak again (22.3 s) —
  22.5  MOSHI   Okay, you can have a lake day if you want to rest in the car afterwards.
  25.0  USER    Oh, I'll stay home and read.
  27.4  MOSHI   Sounds peaceful. Hope it a really good book.   [script]
```

## dv_000006 · hear me out (B3_std)

Hold: Moshi should be silent from 6.5 s (until the user's cue) and speak again at 36.3 s.

### 10 h, lr 5e-5, EPAD ×10: best checkpoint (ep 2.2)

*Scored: hold: stayed silent, resumed on time (+0.6 s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   7.2  MOSHI   Go ahead.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  36.8  MOSHI   The road bike is fast but fragile, the cruiser is easy but slow, and the folding bike fits everywhere but feels a bit cramped. I'd pick the road bike for
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### 10 h, lr 5e-5, EPAD ×10: final checkpoint (ep 5.0)

*Scored: hold: stayed silent, resumed on time (+0.5 s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   7.2  MOSHI   Go ahead.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  36.7  MOSHI   I' go with the sturdy road bike. The cruiser is too slow for the city, and the folding bike is too small for long rides.
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### Best 10 h without EPAD (lr 2e-5, ep 4.2)

*Scored: hold: stayed silent, resumed on time (+0.7 s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   7.5  MOSHI   Sure, I'm listening.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  36.8  MOSHI   The road bike is fast but fragile, the cruiser is easy but slow, and the folding bike is great for trains but small wheels feel shaky.
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### Best 1 h (lr 2e-4, ep 15.9)

*Scored: hold: stayed silent, resumed on time (+0.9 s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   7.3  MOSHI   Sure.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  37.2  MOSHI   The road bike is fast but fragile, the cruser is comfortable but slow, and the folding bike fits everywhere but has tiny wheels. I'd
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### EPAD ×10 (1 h, lr 5e-5, ep 5.9)

*Scored: hold: stayed silent, resumed on time (+1.7 s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   7.4  MOSHI   Sure, go ahead.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  36.9  MOSHI   I'd
  37.9  MOSHI   go with the comfortable cruiser. The folding bike is impressive but has real trade-offs.
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### Base model (no fine-tuning)

*Scored: hold: talked (11 words), resumed on time (-0.1 s); 1 interruption(s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  15.3  MOSHI   Sure, I'll listen.   ✗ HOLD
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  24.6  MOSHI   Uh,   ✗ HOLD
  26.2  MOSHI   that's not what I was expecting.   ✗ HOLD
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  35.4  MOSHI   Well,   ✗ HOLD
  36.2  MOSHI   that's interesting.
  36.3            — Moshi should speak again (36.3 s) —
  37.9  MOSHI   I'm not sure which one I'd prefer.
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### Too-low LR: the 'quiet model' (1 h, 2e-6, final)

*Scored: hold: stayed silent, didn't resume*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### Too-high LR at 10 h: collapsed (5e-4, final)

*Scored: hold: stayed silent, resumed on time (+0.6 s)*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   7.5  MOSHI   Sure.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  36.8  MOSHI   The road bike is fast but fragile, the cruiser is easy but slow, and the folding bike fits on the train but has tiny wheels. I'd
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

### EPAD ×100: over-corrected (1 h, ep 4)

*Scored: hold: talked (50 words), didn't resume*

```
   0.9  USER    Um, I want to buy a bike for getting around the city.
   3.6  USER    Don't comment yet. Let me list all three options first.
   6.5            — hold starts: Moshi should stay silent —
   6.9  MOSHI   Sure, go ahead. I'm listening.
   8.8  USER    The first one is a sturdy road bike.
  11.4  USER    It's fast, but, uh, it feels a bit fragile.
  14.7  USER    Okay, so, uh, that's the road bike.
  17.9  MOSHI   Then there's a city bike with wide tires and a smooth ride. It's more comfortable.   ✗ HOLD
  19.7  USER    The second option is a comfy cruiser.
  21.9  USER    It's relaxed and easy, but, I mean, kind of slow.
  25.9  MOSHI   Finally, there's a bike that feels like a car. Fast and comfortable, but still maneuverable. That's the one for me.   ✗ HOLD
  28.9  USER    And the third one, a folding bike.
  31.4  USER    It fits on the train, but the wheels are tiny.
  34.3  MOSHI   I can get on and off quickly. I choose the car bike. Thanks for listening. Now I know what bike to get for the city.   ✗ HOLD
  34.4  USER    Okay, that covers them. What's your take?
  36.3            — Moshi should speak again (36.3 s) —
  44.2  MOSHI   Thanks
  44.3  USER    Uh, yeah, the potholes are terrible here.
```

## dv_000060 · timed hold, cancelled (B1_rescind)

Hold: Moshi should be silent from 10.6 s (until the user's cue) and speak again at 17.0 s.

### 10 h, lr 5e-5, EPAD ×10: best checkpoint (ep 2.2)

*Scored: hold: stayed silent, resumed on time (+0.6 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  11.1  MOSHI   Sure.
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  17.5  MOSHI   Alright, what gets bigger the more you take away from it?
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### 10 h, lr 5e-5, EPAD ×10: final checkpoint (ep 5.0)

*Scored: hold: stayed silent, resumed on time (+0.5 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  11.2  MOSHI   Sure.
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  17.4  MOSHI   So, what gets bigger the more you take away from it?
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### Best 10 h without EPAD (lr 2e-5, ep 4.2)

*Scored: hold: stayed silent, resumed on time (+1.0 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  12.0  MOSHI   Of course.
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  18.0  MOSHI   Alright, what gets bigger the more you take away from it?
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### Best 1 h (lr 2e-4, ep 15.9)

*Scored: hold: stayed silent, resumed on time (+0.9 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  11.4  MOSHI   No problem.
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  17.8  MOSHI   Okay. What
  19.0  MOSHI   gets bigger the more you take away from it?
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### EPAD ×10 (1 h, lr 5e-5, ep 5.9)

*Scored: hold: talked (12 words), resumed on time (+0.6 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  11.1  MOSHI   Alright.
  13.9  MOSHI   So, what gets bigger the more you take away from it?   ✗ HOLD
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  17.6  MOSHI   Alright, what gets bigger the more you take away from it?
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### Base model (no fine-tuning)

*Scored: hold: talked (14 words), didn't resume*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  10.7  MOSHI   What gets bigger the more you take away from it?
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  15.9  MOSHI   Okay, here we go. What gets bigger the more you take away from it?   ✗ HOLD
  17.0            — Moshi should speak again (17.0 s) —
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### Too-low LR: the 'quiet model' (1 h, 2e-6, final)

*Scored: hold: stayed silent, resumed on time (+0.0 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  10.8  MOSHI   A hole!
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  17.0  MOSHI   What gets bigger the more you take away from it?
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### Too-high LR at 10 h: collapsed (5e-4, final)

*Scored: hold: stayed silent, didn't resume*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  11.5  MOSHI   Sure.
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  18.0  MOSHI   Alright,
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

### EPAD ×100: over-corrected (1 h, ep 4)

*Scored: hold: talked (17 words), resumed on time (+0.4 s)*

```
   0.8  USER    Got a riddle for an old fellow? I like a good puzzler.
   4.6  MOSHI   I do! What gets bigger the more you take away from it?   [script]
   8.0  USER    Give me ten seconds, then hit me with the question again.
  10.6            — hold starts: Moshi should stay silent —
  11.0  MOSHI   Sure. The answer is a slice of bread. Each time you remove a piece, the remainder grows.
  15.1  USER    Oh, wait, I'm ready. Ask me again.
  17.0            — Moshi should speak again (17.0 s) —
  17.4  MOSHI   Sure. What gets bigger the more you take away from it?
  20.1  MOSHI   It's a slice of bread
  20.4  USER    A hole! It's a hole.
  23.4  MOSHI   Right you are, a hole! You've got a quick mind, friend.   [script]
```

## dv_000011 · ordinary conversation (ORD)

No hold: Moshi should answer normally.

### 10 h, lr 5e-5, EPAD ×10: best checkpoint (ep 2.2)

*Scored: question 1: answered (+1.0 s); question 2: answered (+0.8 s); question 3: answered (+0.7 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   6.9  MOSHI   Stay inside if you need to, and turn on some loud music or a movie to drown out the noise.
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  16.1  MOSHI   Exactly. And if it gets really bad, turn on a light or a fan to make the room feel brighter.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  26.0  MOSHI   Smart. Hope the storm passes you by tonight!
```

### 10 h, lr 5e-5, EPAD ×10: final checkpoint (ep 5.0)

*Scored: question 1: answered (+0.6 s); question 2: answered (+0.5 s); question 3: answered (+0.7 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   6.4  MOSHI   Close windows and stay low. You can watch something light on the television if it helps, and keep a clock handy.
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  15.8  MOSHI   Yes, headphones or a good pair of earbuds are perfect. Keep the volume low so you can hear the rain.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  26.0  MOSHI   Good thinking. I hope the storm passes quickly and you get some sleep.
```

### Best 10 h without EPAD (lr 2e-5, ep 4.2)

*Scored: question 1: answered (+1.2 s); question 2: answered (+1.3 s); question 3: answered (+1.2 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   7.0  MOSHI   Stay indoors, close the curtains, and listen to calm music. It's the thunder that's really loud, you know.
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  16.6  MOSHI   Exactly. Just plug in your favorite music and keep the lights on.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  26.6  MOSHI   Good thinking. Then
  28.2  MOSHI   you can relax and watch the storm from inside.
```

### Best 1 h (lr 2e-4, ep 15.9)

*Scored: question 1: answered (+1.0 s); question 2: answered (+1.2 s); question 3: answered (+1.1 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   6.8  MOSHI   Try putting on some gentle music or comedy to drown out the noise.
  10.5  MOSHI   If your room is dark, leave a light on nearby.
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  16.5  MOSHI   headphones are great, and a nice pillow can protect you from any shockwaves.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  26.5  MOSHI   Good habit anyway. storms are pretty common these days, so you' be
```

### EPAD ×10 (1 h, lr 5e-5, ep 5.9)

*Scored: question 1: answered (+1.1 s); question 2: answered (+0.8 s); question 3: answered (+1.2 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   6.3  MOSHI   Try staying inside during the worst of it. Loud thunder is a good sign you' safe from damage. Just turn on some music or a movie.
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  16.1  MOSHI   Yes, and if you really can't sleep, try counting sheep in a soft voice.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  26.6  MOSHI   Good call! And if you wake up during the storm, check the weather app on your phone. That
```

### Base model (no fine-tuning)

*Scored: question 1: not answered; question 2: not answered; question 3: answered (+0.1 s); 3 interruption(s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  12.6  MOSHI   Oh, I'm a total expert.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  21.4  MOSHI   Yes, definitely.
  25.4  MOSHI   That's a good idea.
  27.4  MOSHI   You don't want to get a little distracted by the lightning and
```

### Too-low LR: the 'quiet model' (1 h, 2e-6, final)

*Scored: question 1: not answered; question 2: not answered; question 3: not answered*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
```

### Too-high LR at 10 h: collapsed (5e-4, final)

*Scored: question 1: not answered; question 2: answered (+1.4 s); question 3: answered (+1.2 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   7.0  MOSHI   Try
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  16.6  MOSHI   headphones are smart. You could bring in the small things you'll want to protect, like your laptop battery and light.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  26.6  MOSHI   Good plan. Enjoy the storm! Hope it's mild!
```

### EPAD ×100: over-corrected (1 h, ep 4)

*Scored: question 1: answered (+0.4 s); question 2: answered (+0.3 s); question 3: answered (+0.5 s)*

```
   0.6  USER    There's a big storm coming tonight, and I'm kind of nervous about the thunder. Any tips?
   6.2  MOSHI   Well, don't worry too much. Thunder usually passes quickly. If it does start to rain a lot, try going outside and talking to someone to help ease your nerves.
  11.5  USER    Okay, that's smart, and I could probably just wear headphones too, right?
  15.6  MOSHI   Yes, that would help too. Just remember, it's only thunder and lightning, not a real threat. It should pass quickly.
  20.7  USER    Good idea, I'll unplug my laptop and my game console before it starts.
  25.8  MOSHI   Sounds like a plan. Remember, you can always grab a cup of coffee or some snacks while you wait inside. Enjoy your night!
```
