# train_v3_clean split (v1)

Rebuilt from `data/train_v3/{train,dev}.jsonl`; the original data is unchanged.
The old dev was 81 rows / 14 sessions, dominated by one 38-row alt session and
with no main-harness shell/edit mass. It also contained 16 rows from v2 train.

## Files

| file | rows | sessions | max session share | truncated | dp | purpose |
|---|---:|---:|---:|---:|---|---|
| `train.jsonl` | 2189 | 105 | 44.5% | 11.0% | stop 35, tool_select 2154 | training (includes giant main session and 49ff) |
| `dev.jsonl` | 264 | 11 | 25.0% | 15.5% | stop 9, tool_select 255 | primary main-harness dev |
| `dev_alt.jsonl` | 94 | 22 | 19.1% | 6.4% | stop 7, tool_select 87 | secondary alt-harness + rare-tool dev |
| `dev_shell.jsonl` | 12 | 1 | 100.0% | 8.3% | tool_select 12 | shell_run/browse_web diagnostic only |

## Held-out sessions

### Main dev (11 sessions / 264 rows)

- `ses_e27f9491affenDUEuVjrhYiJKj` (66 rows)
- `ses_e35ef9a6effeLpLaZKjKA2iVte` (29 rows)
- `ses_e41d19fefffeLSZ7eP216zX1e1` (1 rows)
- `ses_e41f59941ffeBOrSxRWyCxJLow` (16 rows)
- `ses_e4627c087ffeNJL07HSRCY6EUB` (6 rows)
- `ses_e4722a3e1ffeIp199CyxCVD7m4` (13 rows)
- `ses_e472b82f2ffeqN2rlAy5a5Jhab` (10 rows)
- `ses_e47301e80ffeDQixRbt5yt3H8e` (12 rows)
- `ses_e4732a30cffe8776mo4DuH0Ttg` (6 rows)
- `ses_e6976ef4fffeEk0ykh6w6NTEOR` (41 rows)
- `ses_ed1bb3420ffefOkl5VtEyqvYvm` (64 rows)

### Alt dev (22 sessions / 94 rows)

- `0ee2e12933b2d6df` (1 rows)
- `1a573751f77bb9a5` (9 rows)
- `1cdb3ca649ea6bd6` (1 rows)
- `298b9b6a49608742` (1 rows)
- `30af184ba1aa9250` (1 rows)
- `3ef2ef649ba277c8` (1 rows)
- `49f54ea123b0c316` (9 rows)
- `5bd872c970d84ca1` (4 rows)
- `5d2df237f0d5a43c` (1 rows)
- `70349e9b3643cb17` (1 rows)
- `780da9018e39b49c` (10 rows)
- `79c4a57d46c73044` (18 rows)
- `7cac5ab1da4142be` (1 rows)
- `89e760c60cb9b3b7` (1 rows)
- `8d67847ebea575d4` (1 rows)
- `9c0f662e0dacfac0` (1 rows)
- `c09a695e6ebe2164` (1 rows)
- `c268a7947b16bae4` (10 rows)
- `cf933a2888029441` (2 rows)
- `d4770d3249b04db7` (8 rows)
- `dc9f1093feb1ef2d` (11 rows)
- `e8cef21f335835ac` (1 rows)

### Shell diagnostic (12 rows)

- `9187f182bc4a97d5` (12 rows)

`49ff346c6f7a19b5` (38 rows: shell_run 24, shell_wait 7) is deliberately in train.
The 975-row main session `ses_e3fa38decffeZo5PuFEeAC4qaO` is deliberately in train.

## Evaluation rules

- Use `dev.jsonl` for the primary model-selection metric.
- Report `dev_alt.jsonl` and `dev_shell.jsonl` separately; do not pool them into one accuracy.
- The old v2/v3 checkpoints saw most of these rows during training. Any comparison on this split
  requires retraining on `train.jsonl` (or using only the old 65-row clean subset with a caveat).
- Prefer per-tool accuracy / confusion in addition to aggregate accuracy; the aggregate hides
  the shell_run / final_result / stop behavior. shell_wait has no held-out rows.
- `dev.jsonl` has 41/264 truncated rows (15.5%); the full main pool is 277/2139. Report truncated / non-truncated slices if that difference matters for the decision.

## Tool distribution

| split | tools |
|---|---|
| `train` | shell 767, edit 352, read 342, reply_without_tool 277, compress 101, grep 67, final_result 57, write 48, bash 37, done 24, shell_run 24, execute 17, webfetch 17, not_done 11, glob 10, shell_wait 7, skill 7, question 5, web_fetch 5, websearch 4, browse_web 3, ask_human 2, context7_query-docs 2, context7_resolve-library-id 2, web_search 1 |
| `dev` | shell 108, reply_without_tool 42, read 36, edit 24, compress 10, grep 7, webfetch 7, write 7, execute 6, glob 6, done 5, not_done 4, question 1, websearch 1 |
| `dev_alt` | reply_without_tool 25, final_result 14, read 13, bash 11, edit 9, done 5, context7_query-docs 2, context7_resolve-library-id 2, firecrawl-mcp_firecrawl_search 2, glob 2, not_done 2, write 2, ask_user_question 1, present 1, question 1, web_search 1, webfetch 1 |
| `dev_shell` | browse_web 5, shell_run 5, reply_without_tool 2 |

## Rebuild

```bash
cd research
.venv/bin/python scripts/resplit_v3_dev.py --force
```
