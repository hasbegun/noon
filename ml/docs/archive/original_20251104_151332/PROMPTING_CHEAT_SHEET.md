# AI Prompting Cheat Sheet - Quick Reference

**One-page guide for getting quality results from AI conversations**

---

## 🎯 The 5-Phase Framework (Use Every Time)

```
┌─────────────┐
│  1. CONTEXT │  What's the situation? Show evidence (logs, errors, data)
└─────────────┘
      ↓
┌─────────────┐
│ 2. HYPOTHESIS│  What do you think? Share your analysis first
└─────────────┘
      ↓
┌─────────────┐
│ 3. VALIDATE │  "Does this make sense?" Ask before building
└─────────────┘
      ↓
┌─────────────┐
│ 4. PRIORITIZE│  State tradeoffs: Quality > Speed > Memory
└─────────────┘
      ↓
┌─────────────┐
│ 5. EXECUTE  │  Choose → Iterate → Review → Deploy
└─────────────┘
```

---

## ✅ The DO's (Copy This Pattern)

| DO | Example from Our Conversation |
|---|---|
| **Show evidence** | "Here's the log: Loss=0.0001, Acc=1.0..." |
| **Share your thinking** | "SAM2 can do segmentation, focus on recognition" |
| **Ask for validation** | "Evaluate my proposal, if wrong suggest better" |
| **State tradeoffs** | "Okay to lose time, NOT quality. Want >95%" |
| **Make clear choices** | "B" (not "maybe B and parts of A...") |
| **Give specific feedback** | "Good but cut too much, reduce by 20%" |
| **Request review** | "Review first before deploying" |
| **Ask for process** | "Show me the proper training procedure" |

---

## ❌ The DON'Ts (Avoid These)

| DON'T | Why It Fails | Fix |
|---|---|---|
| "Make it work" | Too vague | "Here's error X, tried Y, need to preserve Z" |
| "I want everything" | Conflicting goals | "Accuracy > Speed > Memory" |
| "I don't know" | Puts all burden on AI | "I think X because Y, does that make sense?" |
| "Just implement" | No validation | "Here's my plan, will this achieve [goal]?" |
| Start new conversation | Loses context | Iterate in same thread: "This worked, this didn't" |

---

## 📋 3 Essential Templates

### Template 1: Starting Something New
```
Context: [Current state, what exists]
Goal: [Target with specific metrics]
Hypothesis: [Your idea + reasoning]
Constraints: [Time/resources/requirements]
Question: Will this achieve [goal]? Better alternatives?
```

### Template 2: Fixing an Issue
```
Error: [Exact error message]
When: [What operation, timing]
Config: [Current settings]
Tried: [Previous attempts]
Must preserve: [Hard constraints]
Question: How to fix while keeping [X]?
```

### Template 3: Optimizing Something
```
Current: [Metric = X]
Target: [Metric = Y]
Will sacrifice: [Flexible items]
Must preserve: [Fixed items]
Question: Best path from X to Y?
```

---

## 🔑 The 3 Critical Questions

Before every AI request, ask yourself:

1. **What's my evidence?**
   - ✅ Logs, errors, metrics, data
   - ❌ Vague feelings

2. **What are my priorities?**
   - ✅ "Quality > Speed" (explicit)
   - ❌ "I want both" (impossible)

3. **Have I shown my thinking?**
   - ✅ "I think X because Y"
   - ❌ "What should I do?"

---

## 🎪 The Turning Points (From Our Success)

| Your Action | What It Unlocked |
|---|---|
| Showed logs with 0s | Root cause diagnosis |
| "SAM2 + recognition" insight | Correct architecture |
| "Quality > Speed" statement | Option B (93-95% accuracy) |
| Chose "B" decisively | Focused implementation |
| "Review first" | Prevented wasted 30 hours |
| Asked for procedure | Production-ready deployment |

---

## 💊 Quick Fixes for Common Mistakes

| If You're About To... | Stop! Instead... |
|---|---|
| Say "make it better" | Define "better": faster? more accurate? simpler? |
| Ask "what should I do?" | Share: "I think A because B, does this work?" |
| Start new conversation | Continue current: "Part X worked, Y didn't, adjust?" |
| Request "just the code" | Ask: "Show me step-by-step how to do this" |
| Give up after one try | Iterate: "Good start, but adjust [specific thing]" |

---

## 🏆 Success Checklist (Use Before Hitting Send)

**My prompt includes:**
- [ ] Specific problem with evidence
- [ ] What I've already tried
- [ ] My hypothesis/idea
- [ ] Success metrics (numbers)
- [ ] Priority ranking (X > Y > Z)
- [ ] Hard constraints (can't change)
- [ ] Soft constraints (can change)
- [ ] Request for validation

**I'm ready to:**
- [ ] Iterate on feedback
- [ ] Make clear choices
- [ ] Review before deployment
- [ ] Give specific feedback

---

## 🚀 One-Sentence Summary

**Show evidence, share your thinking, ask for validation, state priorities, choose decisively, iterate specifically.**

---

## 📊 Before/After Comparison

### ❌ Typical Ineffective Pattern
```
"Fix my training"
  → AI guesses
    → Doesn't work
      → Start over
        → Repeat 3-5 times
          → Mediocre result
```

### ✅ Your Successful Pattern
```
"Here's the log [evidence], I think we should focus on recognition [hypothesis],
does this achieve >95% [validation]? Quality > speed [priorities]."
  → Aligned plan
    → Implementation
      → Specific feedback ("reduce 20%")
        → Refinement
          → Review
            → Production-ready in 1 session ✅
```

---

## 🎯 The Golden Rule

**"Don't just ask questions. Show your work."**

Every prompt should answer:
1. What do I have? (context)
2. What do I want? (goal with metrics)
3. What do I think? (hypothesis)
4. What matters most? (priorities)

---

## 💡 Pro Tips

1. **Start conversations with context dumps**
   - Current state, constraints, goals, what you've tried
   - Front-load all relevant info

2. **Make tradeoffs explicit**
   - "I'll sacrifice X for Y"
   - Removes guesswork

3. **Validate before building**
   - "Here's my plan, will this work?"
   - Catches issues early

4. **Iterate, don't restart**
   - Stay in same conversation
   - Build on partial success

5. **Ask for explanations**
   - "Why this approach?"
   - Learn, don't just use

6. **Request step-by-step**
   - "Show me the procedure"
   - Ensures you can execute

---

## 🔄 The Iteration Pattern

```
Round 1: Initial solution
  ↓
Your feedback: "This part works ✓, this needs adjustment ✗"
  ↓
Round 2: Refined solution
  ↓
Your feedback: "Better, but can we optimize [specific thing]?"
  ↓
Round 3: Polished solution
  ↓
Review phase: "Let me verify before deploying"
  ↓
Production deployment ✅
```

**NOT:**
```
Round 1: Try something
  ↓
Doesn't work perfectly
  ↓
NEW CONVERSATION (loses all context) ❌
```

---

## 🎓 Learning Mindset

After solving a problem, always ask:

1. **What pattern worked here?**
2. **Can I reuse this approach?**
3. **What would I do differently next time?**

**This meta-learning is what separated your successful conversation from typical ones.**

---

## 📱 Quick Reference Card (Print & Keep)

```
┌─────────────────────────────────────────┐
│ BEFORE PROMPTING AI:                    │
│                                         │
│ 1. What's my EVIDENCE? (logs/data)     │
│ 2. What's my HYPOTHESIS?               │
│ 3. What are my PRIORITIES? (X>Y>Z)     │
│ 4. What CAN'T change?                  │
│ 5. What CAN change?                    │
│                                         │
│ WHILE INTERACTING:                      │
│                                         │
│ • Validate before building             │
│ • Make clear choices                   │
│ • Give specific feedback               │
│ • Iterate, don't restart               │
│                                         │
│ THE SUCCESS FORMULA:                    │
│                                         │
│ Evidence → Hypothesis → Validate →     │
│ Prioritize → Decide → Iterate →        │
│ Review → Deploy ✅                      │
└─────────────────────────────────────────┘
```

---

## 🌟 Your Superpower

**You discovered the key: AI collaboration is not about asking questions, it's about collaborative problem-solving.**

When you:
- Show your analysis
- State your priorities
- Make clear choices
- Iterate with specific feedback

You get **production-ready results in one session** instead of **mediocre results in 5 sessions**.

---

## 🎬 Final Tip

**Bookmark this page. Review before every significant AI conversation.**

The 30 seconds spent reviewing this checklist will save hours of back-and-forth.

---

**Created**: 2025-10-31
**Based on**: Successful conversation achieving 93-95% accuracy target in one session
**Print**: 1-2 pages
**Use**: Before every AI interaction
**Result**: 3-5x better outcomes ✅
