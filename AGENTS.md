## QUICK START

Read DEVELOPER_GUIDE.md to understand how the codebase works when starting from scratch with no context 

## TOOLS
- Use exa code search tool to ensure you are using the most up-to-date accurate code during your plans before coding
- Use exa web search if you need to access the internet.
- Use package search to find EXACT types/functions/specifics of packages we work with if exa does not suffice.

## IMPORTANT RULES
- never use dynamic imports (unless asked to) like `await import(...)`
- never cast to `any`
- do not add extra defensive checks or try/catch blocks
- never EVER run a server using npm run dev ONLY if the user asks you too
- use exa code search tool to verify your coding plans BEFORE coding
- do not create alternative fallbacks unless the user asks for it. the user only wants one solution, if that solution doesn't work then the code simply won't work

## NOTES
- you may be working with multiple other instances at the same time, do not worry if the code is changed on you on the spot or a lint error occurs mid build, that may be because an agent is currently mid progress. they're supposed to work with each other tho so don't trip.

## GIT
If you create any new files which should be added to version control, please tell me after you are done coding. If you run git diff and notice any files you didn't edit that were changed, this is entirely normal. There are many agents acting in parallel on different parts of the code, so it was probably another agent.