#!/bin/sh

apt install htop
apt install nano

apt install dos2unix
find . -type f -print0 | xargs -0 dos2unix