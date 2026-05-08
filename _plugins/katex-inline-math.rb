# frozen_string_literal: true

# Upgrade single-dollar inline math to kramdown-compatible double-dollar math.
#
# Why: kramdown only recognises `$$ ... $$` as math. The Chirpy theme's MathJax
# config (assets/js/data/mathjax.js) papers over this by also scanning the
# rendered page for `$ ... $` in the browser. With server-side rendering
# (jektex/KaTeX) there is no client-side scanner, so single-dollar inline math
# would be left as literal text.
#
# This pre-render hook rewrites `$ ... $` to `$$ ... $$` in the raw Markdown so
# kramdown emits `\( ... \)`, which jektex then renders to KaTeX HTML at build
# time.
#
# Scope and limitations:
#   * Only runs on documents/pages whose front matter sets `math: true`
#     (the same flag Chirpy uses to opt a page into math rendering).
#   * Operates on raw Markdown before fenced code blocks are isolated, so it
#     deliberately skips lines that look like fenced-code delimiters and never
#     converts across newlines. Avoid bare `$` inside code on math pages, or
#     escape it as `\$`.
#   * Leaves `$$ ... $$`, `\$`, and `\( ... \)` / `\[ ... \]` untouched.

module KatexInlineMath
  # `$expr$` where:
  #   * the opening `$` is not preceded by `$` or `\` (skip `$$`, `\$`)
  #   * the opening `$` is not followed by `$` (skip `$$`)
  #   * the body has no `$` or newline; `\$` escapes are allowed
  #   * the closing `$` is not followed by `$` (skip `$$`)
  INLINE_MATH = /(?<![\\$])\$(?!\$)((?:\\.|[^$\n])+?)\$(?!\$)/.freeze

  FENCE = /\A\s*(```|~~~)/.freeze

  module_function

  def enabled?(doc)
    doc.respond_to?(:data) && doc.data.is_a?(Hash) && doc.data["math"] == true
  end

  def convert(content)
    in_fence = false
    content.each_line.map do |line|
      if line.match?(FENCE)
        in_fence = !in_fence
        line
      elsif in_fence
        line
      else
        line.gsub(INLINE_MATH) { "$$#{Regexp.last_match(1)}$$" }
      end
    end.join
  end
end

Jekyll::Hooks.register [:documents, :pages], :pre_render do |doc|
  next unless KatexInlineMath.enabled?(doc)

  doc.content = KatexInlineMath.convert(doc.content)
end
