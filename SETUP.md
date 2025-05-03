# Local Setup Guide

This guide will help you run the Jekyll site locally on your computer.

## Prerequisites

1. Ruby (recommended version: 2.7.x)
2. Bundler (gem install bundler)

## Setup Instructions

### For macOS Users

1. First, clean any existing bundler setup:
```bash
rm -rf vendor/
rm -rf .bundle/
rm -f Gemfile.lock
```

2. Install dependencies:
```bash
bundle install
```

3. Run the Jekyll server:
```bash
bundle exec jekyll serve
```

### Troubleshooting Common Issues

#### FFI Error on macOS

If you encounter an error related to `ffi_c` or `sassc` failing to build native extensions:

1. Install Xcode Command Line Tools:
```bash
xcode-select --install
```

2. Install Ruby using a version manager like rbenv:
```bash
# Install rbenv
brew install rbenv

# Setup rbenv
rbenv init

# Install Ruby 2.7.4
rbenv install 2.7.4
rbenv global 2.7.4
```

3. Reinstall bundler and dependencies:
```bash
gem install bundler
bundle install
```

#### Jekyll Build Errors

If you see Jekyll build errors:

1. Make sure you have the correct dependencies installed:
```bash
bundle update
```

2. Check for YAML front matter in all markdown files:
- All markdown files should have `---` at the beginning and end of the front matter
- There should be no tabs in the front matter, only spaces

## Running the Site

After successful installation, run:

```bash
bundle exec jekyll serve
```

This will start a local server at http://localhost:4000/tensor-programming/ where you can preview your site. 