# Source for blog https://robotchinwag.com

<!-- * Website CMS/editor: https://app.pagescms.org/joshlk/robotchinwag.com -->
* Analytics at: https://robotchinwag.goatcounter.com/
* Comments stored using [utteranc](https://utteranc.es/) at repo: https://github.com/joshlk/robotchinwag_comments
* Use [Typora](https://typora.io/) for editing markdown and VSCode otherwise

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

Dev, cd to dir:
```bash
chruby 3.1.3
bundle exec jekyll s
```

Use [this guide]([img_path](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide#upgrade-from-starter)) update the theme version.

To update the template file:
```bash
git remote add template git@github.com:cotes2020/chirpy-starter.git
git merge template/master --squash
```

## Modifications to theme

You can make modifications to the site by copying files from the [themes repo](https://github.com/cotes2020/jekyll-theme-chirpy) and modifying them.

## Licence

The theme is [MIT licensed]([/LICENSE](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/LICENSE). The blog contents are copyrighted by Josh Levy-Kramer. All rights are reserved.
