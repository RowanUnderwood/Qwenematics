# H3 community discoveries (from ComfyUI-Fantastic-MiniMaxH3-PromptBuilder guide, Part A §6)

6Community Discoveries#
The material in this section is not part of MiniMax's published guide. It comes from tags the community has found to work in practice. Treat it as a starting point rather than a specification: behaviour may vary between checkpoints, and any of it may stop working in a future model revision.
6.1Audio and Performance Tags#
These tags go inside a <d> block, alongside the spoken text, to shape delivery — pauses, breathing, emphasis, and non-verbal sounds. They are distinct from the <scenetrans> and <cutoff> markers in 4.4, which control how a line spans a cut. Most are self-closing; the few that wrap text are shown with a closing tag.

TagWhat it doesExample

<pause>Short pauseOkay, so. <pause> This is just me talking.

<long pause>Longer pauseI mean… <long pause> I don't even know.

<breath>Breathing soundAnd then… <breath> it just happened.

<inhale> / <exhale>In/out breath<inhale> Alright, let's do this.

<catches breath>Out of breathWait… <catches breath> hold on a sec.

<deep breath>Calming down<deep breath> Okay. I can do this.

<i>word</i>Emphasize 1–4 wordsI was <i>not</i> expecting that.

<whisper>…</whisper>Whisper delivery<whisper> Don't tell anyone this.</whisper>

<humming>…</humming>Humming a tune<humming> da-da-da-beautiful-day.</humming>

<laughs> / <chuckle>LaughingThat's… <laughs> that's actually funny.

<sighs>Sigh<sighs> I really tried.

<uh>Filler / hesitationSo, like… <uh> what was I saying?

<stutter>Stutters the words<stutter> I ca can't believe that.

<gasp>Sharp intake<gasp> Oh my God.

<coughs>Cough<coughs> Sorry, one sec.

<clears throat>Throat clear<clears throat> So anyway…

<sniff>Sniffing<sniff> It's just… really sad.

<softer>Quieter delivery<softer> I don't think I can say it.

<mhm>Agreement soundYeah, <mhm> exactly.

<phew>Relief<phew> That was close.

Use them sparingly. One or two per line reads as natural performance; stacking many in a single utterance tends to make delivery choppy and can cause the model to drop the tags entirely.
The prompt builder carries this same list in its Delivery row, grouped into pauses and breath, delivery and emphasis, and non-verbal sounds. Hovering the picker previews the example; the four tags that wrap text wrap the current selection when there is one.
