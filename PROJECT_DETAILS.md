# What This Project Is

A plain-language explanation of `python-dpo` — what it does, the thinking behind it, and
where it honestly stands. No mathematics.

For *how* each stage is built, see the numbered reports (`02_PROBLEM_DATASET.md` through
`12_PIPELINE_ORCHESTRATION_AND_PRODUCTIONIZATION.md`). For usage, see `README.md`. This file
is the one to read first.

---

## What it does, in one sentence

It tries to make a small AI coding model better at writing Python — by having the model
write code, running that code against real tests to see what actually works, and then
teaching the model from the difference between its own good and bad attempts.

---

## The problem it's addressing

An AI model that writes code learned to do it by reading enormous amounts of text. It
learned what code *looks like*. Nobody ever checked whether the code it writes actually
**runs**.

That's the gap. The model is optimizing for "this looks like plausible Python," not "this
works." Those are different targets, and the difference is where bugs live.

So how do you teach it the second thing?

---

## The idea

The obvious approach is: hire experts to write correct solutions, and have the model imitate
them. That's expensive, slow, and caps the model at the experts' level.

This project does something else:

1. Give the model a problem
2. Let it write **several different attempts** — five, say
3. **Actually run** each attempt against tests
4. Now you know, objectively, which attempts worked and which didn't
5. Teach the model: *"attempt 3 was better than attempt 1"*

The model becomes its own teacher. The only thing supplied from outside is the problems and
the tests. Nobody has to write correct answers.

**The crucial move is step 5 — teaching by comparison rather than correction.** You're not
saying "here's the right answer." You're saying "between these two things you produced, this
one was better." That turns out to be enough for a model to shift its behavior, and it's far
cheaper to obtain, because a test result gives you the comparison for free.

That technique has a name — DPO, Direct Preference Optimization — but the name doesn't
matter much. The idea is: learn from pairs of your own attempts, ranked by something
objective.

---

## The philosophy

This is the part that explains why the codebase looks the way it does. A few convictions are
baked into it.

### 1. Truth comes from execution, not opinion

"Which attempt was better" is never decided by a human rater or by another AI grading the
work. It's decided by running the code. A test passes or it doesn't. This is the
foundational commitment, and much of the design follows from refusing to compromise it.

### 2. The main danger is fooling yourself

This is the conviction that shapes the most code. When you're measuring whether your own
training worked, there are a dozen ways to accidentally produce a flattering number —
evaluate on problems you trained on, quietly drop the failures, tune the success threshold
after seeing the result, retry until you like the answer.

So the system is full of **refusals**. It will stop and fail rather than produce a number it
can't stand behind:

- It refuses to train when there are no real preference pairs to train on.
- It refuses to evaluate when a training problem has leaked into the held-out test set.
- It refuses to declare success on evidence too thin to support the claim.
- It refuses to write a recommendation that has no evidence attached.
- Its reports say "potential data gap," never "X caused Y," when the evidence only shows
  the two occurred together.

Several of these have fired in practice. They are the system working, not bugs.

### 3. Untrusted code is untrusted

The model writes code, and that code gets run. It could delete files or hang forever — not
maliciously, just because the model made a mistake. So generated code only ever executes
inside a locked-down container: no network, no access to the host filesystem, a non-root
user, and hard limits on time, memory and processes.

### 4. Everything must be traceable

If a trained model turns out good or bad, you need to know exactly what produced it — which
comparisons trained it, which test results produced those comparisons, which attempts
produced those results, which problems produced those attempts. Every step records its
inputs and outputs with checksums, and the full chain is stored as a fact rather than
reconstructed by hand.

### 5. Nothing gets promoted automatically

A trained model is registered as "experimental" and stays there. Marking it as good requires
a human to do so explicitly, backed by a passing evaluation record.

---

## Why it looks disproportionate

Here is the thing most likely to confuse a newcomer: **the machinery is enormous relative to
what it has produced.** Fifty-odd thousand lines of code, twelve stages, over fifteen hundred
tests — and the model it trained learned essentially nothing, from three examples.

That mismatch is real, and it is not a mistake. The project was built **apparatus first**.
The reasoning: a preference-learning pipeline that quietly produces wrong numbers is worse
than useless, so build the thing that can be trusted, prove it works end to end, and only
then feed it real data.

Think of it as building a laboratory before running the experiment. The lab is finished and
verified. The experiment has not really been run.

---

## How the pieces fit together

The pipeline runs as a sequence, each step consuming what the previous one produced:

**Problems** — a curated set of Python tasks, each with a reference solution and tests.

**Generation** — the model writes several attempts at each problem, using varied prompting
styles to encourage different approaches.

**Execution** — each attempt runs against its problem's tests, inside the sandbox. This is
where objective evidence gets created.

**Ranking** — attempts are scored and ordered by how they actually performed.

**Preference pairs** — for each problem, better attempts are paired against worse ones.
Comparisons are only ever made *within* a single problem; comparing an attempt at one
problem against an attempt at another would be meaningless.

**Training** — the model is fine-tuned on those pairs. Only a small adapter is trained, not
the whole model, which is what makes this feasible on a single consumer graphics card.

**Evaluation** — the original model and the tuned model are compared on held-out problems
neither was trained on, generating and executing code for both under identical conditions.

**Analysis** — failures are classified, gaps between what was trained and what was measured
are identified, and a proposed next experiment is written out. It stops there; it never
retrains on its own.

**Packaging** — the trained adapter is bundled into something loadable, verified by making
it generate code and running that code, then registered.

A single command runs all of it, and one identifier ties every artifact together.

---

## Where it actually stands

The pipeline works. It has been run end to end on real hardware — real code generation, real
sandboxed execution, real training, real packaging.

**But the experiment cannot produce a meaningful result yet, and the system says so rather
than pretending otherwise.**

The reason is the problem set. There are only ten problems. Of those, the model reliably
succeeds at four and reliably fails at three. Those seven are useless for this technique — if
every attempt is equally good, or equally bad, there is nothing to compare. Only three
problems sit in the zone where the model sometimes succeeds and sometimes fails, and one of
those is reserved for testing.

That leaves roughly two usable problems. Not enough to teach anything.

The analysis stage adds a second finding, and it is arguably sharper: the problems that were
trained on and the problems used for evaluation **share no categories at all**. Nothing that
was trained was ever measured, and nothing that was measured was ever trained.

**The fix is not more code.** It is more problems — going from ten hand-written ones to a few
hundred, so that enough of them sit where the model is genuinely uncertain, and so the
training and evaluation sets can overlap in subject matter. That is a content-writing effort,
not an engineering one.

---

## The honest summary

What exists is a working, trustworthy apparatus for preference-based training of a code
model, built to refuse rather than flatter, and verified end to end.

What does not exist yet is a result. The dataset is too small to produce one, and the
pipeline reports that plainly instead of manufacturing a number.
