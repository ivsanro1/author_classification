#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
jupyter nbextension enable toc2/main
# jupyter nbextension enable execution_dependencies/execution_dependencies

cd $HOME/work; jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''