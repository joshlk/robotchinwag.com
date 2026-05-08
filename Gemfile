# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll-theme-chirpy", "~> 7.4", ">= 7.4.1"

# Server-side KaTeX rendering (replaces client-side MathJax).
# See: https://github.com/cotes2020/jekyll-theme-chirpy/pull/2603
group :jekyll_plugins do
  gem "jektex", "~> 0.1.1"
end

gem "html-proofer", "~> 5.0", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
