# Early Detection of Collective Psychological Pressure

This project studies stress-related linguistic signals and aggregate population-level patterns in online text. It does not diagnose mental illness or determine an individual's clinical or psychological condition.

## Dataset roles

- Dreaddit: primary supervised stress-language task.
- SenticNet stress-detection spreadsheets: external Reddit and Twitter validation; labels remain source-specific.
- GoEmotions: separate multi-label emotion task, not a stress-label substitute.
- Mendeley stress-indicator survey: informs a stressor taxonomy; it is not treated as social-media text.
- TensiStrength lexicons: auxiliary tension signal, not ground truth.

## Current implementation state

Phase 1 scaffold and the initial dataset audit are being established. Source datasets stay in the parent `../datasets/` folder; audit reports, model artifacts, and the Hugging Face cache stay inside this project.

## Project layout

See the directory structure under this project. Run commands from the project root with its `.venv` activated.