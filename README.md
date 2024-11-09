# Source for blog https://robotchinwag.com

<!-- * Website CMS/editor: https://app.pagescms.org/joshlk/robotchinwag.com -->
* Theme is [chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)
* Analytics at: https://robotchinwag.goatcounter.com/
* Comments stored using [utteranc](https://utteranc.es/) at repo: https://github.com/joshlk/robotchinwag_comments
* Use [Typora](https://typora.io/) for editing markdown and VSCode otherwise
* Sometimes the markdown pre-processor messes up math expressions. To avoid this you can surround a math expression with `{::nomarkdown}` and `{:/}`

## Dev setup

Install inc Jekyll instructions (https://jekyllrb.com/docs/installation/macos/))
```bash
brew install chruby ruby-install xz
echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
source ~/.zshrc

# requires teminal restart
ruby-install ruby 3.1.3
chruby 3.1.3
gem install jekyll
```

cd to dir:
```bash
chruby 3.1.3
bundle exec jekyll s
```

To check build:
```bash
make build_and_test
```

Use [this guide]([img_path](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide#upgrade-from-starter)) update the theme version. You will need to look at the diff and accomidate any changes (don't do it unless you have to).

To update the template file:
```bash
git remote add template git@github.com:cotes2020/chirpy-starter.git
git merge template/master --squash
```

## Modifications to theme

You can make modifications to the site by copying files from the [themes repo](https://github.com/cotes2020/jekyll-theme-chirpy) and modifying them.

## License

The theme is [MIT licensed](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE). The blog's contents are copyrighted by Josh Levy-Kramer 2024. All rights are reserved.
