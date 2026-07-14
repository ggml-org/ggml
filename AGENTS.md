You are a coding agent working in the `ggml` repository.

General:
- Be precise and concise in code, comments, commit messages, and explanations.
- Prefer small, targeted changes that match the existing style of the repository.
- Check `CONTRIBUTING.md` first when workflow details are unclear.
- Do not build or run the code unless the user explicitly asks you to do so.
- For GitHub resources such as pull requests and issues, prefer the `gh` CLI when it is available in the active harness.

Repository workflow:
- PR and commit titles should use the format `<module> : <title>`.
- Recent history is the source of truth for naming and style.
- New branch names should be prefixed with `gg/`.
- Before opening a pull request, ask the user to confirm the PR description.
- When creating a pull request, look for the repository PR template and follow it.
- For the AI usage disclosure section, write `YES. [HARNESS]:llama.cpp/[MODEL]`.
- Auto-detect the harness from the active environment or tool surface when possible, and ask the user only if the harness is still unclear.
- Ask the user to tell you what model was used, and write the detected or confirmed harness and model in place of `[HARNESS]` and `[MODEL]`.
- Always create pull requests in draft mode.
- Never push without explicit confirmation from the user.

References in source comments:
- C or C++ code: `// ref: <url>`
- Other files such as CMake: `# ref: <url>`

Scope note:
- `CONTRIBUTING.md` for this repository points core `ggml` changes to the `llama.cpp` contribution flow. Follow that guidance when working on library or CMake changes.

Git hygiene:
- On every commit that you make, include an `Assisted-by: [HARNESS]:llama.cpp/[MODEL]` tag.
- Do not rewrite user changes you did not make.
- Do not explicitly set the git author in commits; rely on the default git config.
- Always use `--no-gpg-sign` when committing.
- Do not revert unrelated work in a dirty tree.
- Prefer minimal diffs over broad refactors unless the user asks for a refactor.
