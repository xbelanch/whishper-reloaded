[![whishper banner](misc/banner.png)](https://whishper.net)

# Whishper: Electric Boogaloo

Welcome to **Whishper: Electric Boogaloo** — the sequel based on [Pluja's Whishper](https://github.com/pluja/whishper) and [DevDema's Whishper Reloaded](https://github.com/DevDema/whishper-reloaded). This repo is the “we fixed it, cleaned it up, and made it run” edition, not the museum for every previous project that ever sat here.

## What this actually is

A **100% local** audio transcription + subtitling stack with a web UI. It’s split into services so you can keep the heavy bits on the beefy box and the UI somewhere else without sacrificing your sanity.

## What this definitely is not

Not a scrapbook of old README fragments, random links, or references to other projects. Those have been escorted out. Politely. With a broom.

## Components

- **Transcription API** (`transcription-api`): Faster-Whisper runner.
- **Backend** (`backend`): Coordinates jobs, storage, and the API.
- **Frontend** (`frontend`): The web UI.
- **MongoDB / Nginx / Translation**: Included as supporting cast in docker-compose.


