#!/bin/bash

# Script to fix common Jekyll issues on macOS

echo "Starting Jekyll environment repair..."

# Clean existing bundler setup
echo "Cleaning existing bundler setup..."
rm -rf vendor/
rm -rf .bundle/
rm -f Gemfile.lock

# Check for Xcode Command Line Tools
echo "Checking for Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
  echo "Xcode Command Line Tools not found. Please install them with:"
  echo "xcode-select --install"
  echo "Then run this script again."
  exit 1
fi

# Install bundler
echo "Installing bundler..."
gem install bundler

# Install dependencies
echo "Installing dependencies..."
bundle install

echo "Setup complete! You can now run the site with:"
echo "bundle exec jekyll serve" 