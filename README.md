# Source for blog https://robotchinwag.com

Analytics at: https://robotchinwag.goatcounter.com/
Comments stored using [utteranc](https://utteranc.es/) at repo: https://github.com/joshlk/robotchinwag_comments

## Dev setup

Install inc Jekyll instructions (https://jekyllrb.com/docs/installation/macos/))
```
brew install chruby ruby-install xz
echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
source ~/.zshrc

# requires teminal restart
ruby-install ruby 3.1.3
chruby 3.1.3
gem install jekyll
```

Dev, cd to dir:
```
chruby 3.1.3
bundle exec jekyll s
```

## Modifications to theme

You can make modifications to the site by copying files from the [themes repo](https://github.com/cotes2020/jekyll-theme-chirpy) and modifying them.

