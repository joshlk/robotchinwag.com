# robotchinwag.com

Personal Jekyll blog built on the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)
theme, consumed as a **gem** (`jekyll-theme-chirpy` in `Gemfile`) — the theme source is *not*
vendored into this repo. Deployed to GitHub Pages via `.github/workflows/pages-deploy.yml`
(Ruby 3.3, `jekyll b` then `htmlproofer`).

## Build & test

```bash
bundle install
JEKYLL_ENV=production bundle exec jekyll b -d _site
bundle exec htmlproofer _site --disable-external \
  --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"
# or: make build_and_test
# dev server: bash tools/run.sh
```

Requires Ruby ≥ 3.1 and < 4.0 (Chirpy pins `~> 3.1`) plus a JS runtime for `execjs`
(Node is fine). On macOS the system Ruby (2.6) is too old — use
`brew install ruby@3.4` and prepend `/opt/homebrew/opt/ruby@3.4/bin` to `PATH`.

## Math rendering: server-side KaTeX (not the theme default)

Chirpy ships **MathJax** rendered client-side. This site replaces it with **KaTeX rendered
at build time** via the [`jektex`](https://github.com/yagarea/jektex) plugin, applying the
approach from [cotes2020/jekyll-theme-chirpy#2603](https://github.com/cotes2020/jekyll-theme-chirpy/pull/2603)
locally (the upstream PR was not merged).

Pieces involved:

| File | Role |
|---|---|
| `Gemfile` | `jektex` in `:jekyll_plugins` |
| `_config.yml` | `math.engine: katex` (read by the include overrides) and a `jektex:` block (cache dir, ignore globs, macros) |
| `_includes/head.html` | **Override** of the theme file — adds the KaTeX stylesheet on `math: true` pages when `math.engine == 'katex'` |
| `_includes/js-selector.html` | **Override** of the theme file — skips the MathJax `<script>` tags when `math.engine == 'katex'` |
| `_plugins/katex-inline-math.rb` | Pre-render hook: rewrites single-dollar inline `$…$` → kramdown's `$$…$$` on pages with `math: true` (skips fenced code blocks). Needed because kramdown only recognises `$$…$$`; the theme's MathJax config previously papered over this in the browser. |
| `assets/css/jekyll-theme-chirpy.scss` | `.katex-display` overflow/padding so wide equations scroll instead of overflowing |
| `.gitignore` | `.jektex-cache/` |

Pipeline at build time:

1. `_plugins/katex-inline-math.rb` (`:pre_render`) — `$x$` → `$$x$$` on `math: true` pages.
2. kramdown converts `$$x$$` → `\(x\)` (inline) / `\[x\]` (block) in the rendered HTML.
3. `jektex` (`:post_render`) — replaces `\(…\)` / `\[…\]` with KaTeX HTML using a bundled
   KaTeX JS bundle via `execjs`. Caches results in `.jektex-cache/`.
4. The KaTeX **stylesheet** (CDN, `katex@0.16.9`, with SRI hash) is what makes the markup
   render correctly — keep its version aligned with the KaTeX bundled inside `jektex`.

Switching back to MathJax is a one-line change: set `math.engine:` to `mathjax` (or empty)
in `_config.yml`. The include overrides preserve the original MathJax branch.

### Authoring rules for math posts

- Set `math: true` in front matter (per Chirpy convention).
- Block math: `$$ … $$` on its own lines with blank lines around it.
- Inline math: either `$…$` or `$$…$$` — both work; the plugin normalises the former.
- Don't use `\label` / `\eqref` / `\tag` — KaTeX doesn't support them.
- Avoid a bare `$` inside fenced/inline code on `math: true` posts; escape it as `\$` if
  you must (the plugin skips fenced blocks, but inline code is not parsed).

## Updating the Chirpy theme version

`_includes/head.html` and `_includes/js-selector.html` are **vendored copies** of the
theme's templates with a small KaTeX patch. They shadow the gem's versions, so when you
bump `jekyll-theme-chirpy` you must re-sync them or you'll silently miss upstream fixes.

Procedure:

1. Bump the version constraint in `Gemfile`, run `bundle update jekyll-theme-chirpy`, and
   note the resolved version (`bundle show jekyll-theme-chirpy` or check `Gemfile.lock`).
2. Fetch the new upstream templates for that tag:
   ```bash
   V=v7.6.0   # the resolved version
   curl -sL https://raw.githubusercontent.com/cotes2020/jekyll-theme-chirpy/$V/_includes/head.html        -o /tmp/head.html
   curl -sL https://raw.githubusercontent.com/cotes2020/jekyll-theme-chirpy/$V/_includes/js-selector.html -o /tmp/js-selector.html
   ```
3. Diff them against the local overrides to see what upstream changed:
   ```bash
   diff /tmp/head.html        _includes/head.html
   diff /tmp/js-selector.html _includes/js-selector.html
   ```
4. Replace the local files with the fresh upstream copies, then **re-apply the two KaTeX
   patches** (each is marked by a `{% comment %}` header at the top of the file noting the
   upstream version it tracks — update that version string too):
   - `head.html`: the `{% if page.math and math_engine == 'katex' %}` block that links the
     KaTeX CSS, inserted just before `<!-- Scripts -->`.
   - `js-selector.html`: the `{% assign math_engine … %}{% if math_engine == 'mathjax' %}`
     wrapper around the existing MathJax `<script>` tags inside `{% if page.math %}`.
5. Check whether upstream changed how math is wired (e.g. moved the MathJax block, added a
   `mathjax.js` config option, or — best case — merged native KaTeX support, in which case
   delete these overrides and the plugin and use the theme's config instead).
6. Rebuild and run `htmlproofer`. Spot-check a math-heavy post (e.g.
   `the-tensor-calculus-you-need-for-deep-learning`) and confirm:
   - `.katex` spans present, **zero** `.katex-error` elements,
   - no `MathJax`/`mathjax` script tags,
   - `katex.min.css` linked,
   - no leftover raw `$…$` / `\(…\)` / `\[…\]` outside `<code>`/`<pre>`/`<script>`.
7. If you also bump `jektex`, check what KaTeX version it bundles and update the
   `katex@<version>` CDN URL + SRI `integrity` hash in `head.html`:
   ```bash
   curl -sL https://cdn.jsdelivr.net/npm/katex@<version>/dist/katex.min.css \
     | openssl dgst -sha384 -binary | openssl base64 -A
   ```

## Other local theme customisations

- `assets/css/jekyll-theme-chirpy.scss` — sidebar avatar sizing, hides "recently
  updated"/"trending tags" panel, hides post prev/next nav, mobile topbar title, table
  list wrapping, KaTeX overflow.
- `_includes/comments/` — Giscus comment integration override.
- `_plugins/posts-lastmod-hook.rb` — sets `last_modified_at` from git history.
