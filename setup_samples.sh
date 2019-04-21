#!/bin/bash

current=`pwd`
echo $current
git submodule update -i
git submodule foreach git pull origin master

mkdir ~/workspace
rm ~/workspace/stock_data
ln -s $current/data ~/workspace/stock_data

sh setup/install.sh
sh setup/setup.sh
