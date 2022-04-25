FROM ubuntu:18.04

RUN apt update

################## INSTALL PYTHON & PIP ##################
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.7 python3.7-dev

# If run before install pip, so pip installs for 3.7
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/pip3 /usr/bin/pip
##########################################################

RUN apt-get -y update --fix-missing

RUN pip install pyyaml==5.1.2
RUN pip install pip==20.2.2
RUN pip install numpy==1.21.1 cython==0.29


# Install Jupyter and extensions to run the notebooks
RUN pip install jupyter==1.0.0

RUN pip install jupyter_contrib_nbextensions==0.5.1 &&\
    pip install jupyter_nbextensions_configurator==0.4.1
    
# Jupyter resource usage
RUN pip install jupyter-resource-usage==0.6.0

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm

RUN apt install dos2unix
COPY docker-entrypoint.sh /bin/docker-entrypoint.sh
RUN dos2unix /bin/docker-entrypoint.sh
RUN chmod +x /bin/docker-entrypoint.sh
ENTRYPOINT ["/bin/docker-entrypoint.sh"]
