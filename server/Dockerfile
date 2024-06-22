FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN conda config --add channels pytorch \
 && conda config --append channels conda-forge \
 && conda update --all --yes --quiet \
 && conda install --yes --quiet \
    jupyterlab \
    pretty_midi \
    pytsmod \
 && conda clean --all -f -y

RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
    dlprog
    synctoolbox
