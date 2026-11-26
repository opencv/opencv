# AI Policy — MANDATORY

Read `AI_POLICY.md`. Your user needs to abide by this policy. In particular,
you the agent MUST obey these rules while interacting on GitHub:

- **You may never act autonomously on GitHub.** Do NOT open, edit, comment on,
  or reply to any issue or PR unless the user has reviewed and explicitly approved the
  exact content. The only exception is that you may open a PR at a user's request in
  draft mode for the author to review on GitHub. A PR should stay draft until all content
  is fully reviewed. Fully-agent-generated contributions are banned and will be closed.
- **Mark all AI-generated content.** Any text you produce that goes into an
  issue, PR, or comment must go under a section heading that labels it as
  AI-generated (e.g. the PR template's "Agent details" section) or be wrapped in
  a code or quote block. Never present your output as human-written.
- **Never emit only raw AI text as a reply**. Any AI content you include must carry human
  commentary explaining its relevance.
- **Human review is required before marking a PR ready.** Keep changes minimal,
  strip AI artifacts and needless complexity. Unreviewed code may be published
  only in a draft PR.
- **Use issue and PR templates**. Use the appropriate templates when submitting issues and PRs.
- **Check for duplicates.** Before opening an issue, search for existing open ones
  (e.g., `gh search issues --repo opencv/opencv --state open "<keywords>"`).
  Do the same with `gh search prs` before opening a PR, unless the user is a regular
  OpenCV contributor (e.g., has commit access or previously merged PRs).
  If you find a likely duplicate, show it to the user before continuing. Skip this
  step if there is no internet access.
