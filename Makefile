.PHONY: build_and_test

build_and_test:
	bundle exec jekyll b -d "_site"
	bundle exec htmlproofer _site \
            \-\-disable-external \
            \-\-ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"